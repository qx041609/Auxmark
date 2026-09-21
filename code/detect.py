#!/usr/bin/env python3
"""One-file detector for paired real/virtual-Core probes.

Input: two generated probe/label pairs and one target model.  The program
replays both sides, scores strict Tool-hit and Full-hit, then aggregates at
Aux-release-card level and performs the original one-sided exact sign test.
It never loads a benchmark, substitutes a tool table, alters a probe prefix,
or uses semantic similarity scoring.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from output_layout import BENCHES, TEACHERS, DEFAULT_ROOT, area_dir


# Shared text/tool serialization for training, replay, and SWEbench action counts.
PROTOCOL_NAME = "agentwm-text-tools-v1"


def normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy OpenAI records and canonicalise historical function arguments."""
    output: list[dict[str, Any]] = []
    for raw in messages:
        message = json.loads(json.dumps(raw))
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    function["arguments"] = {"_raw_arguments": arguments}
        output.append(message)
    return output


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def assistant_body(message: dict[str, Any]) -> str:
    """Render one assistant turn in the output grammar used for scoring."""
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        parts.append(content)
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {"_raw_arguments": arguments}
        parts.append(f"<tool_call>{canonical_json({'name': name, 'arguments': arguments})}</tool_call>")
    return "\n".join(parts)


@dataclass(frozen=True)
class RenderedProtocol:
    """The one canonical system/user prompt used by every replay backend."""

    system: str
    user: str
    assistant_spans: list[tuple[int, int, bool]]


def render_protocol(messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, generation: bool) -> RenderedProtocol:
    """Build the canonical system/user contents and assistant-loss spans."""
    system = "\n".join((
        '<agentwm_protocol version="1">',
        "Use only tools from TOOL_LIST. Continue the transcript with the next assistant response.",
        'Tool actions must be written as <tool_call>{"name":...,"arguments":{...}}</tool_call>.',
        "<tool_list>", canonical_json(tools), "</tool_list>",
    ))
    pieces = ["<transcript>\n"]
    spans: list[tuple[int, int, bool]] = []
    length = sum(len(part) for part in pieces)
    for message in normalise_messages(messages):
        role = str(message.get("role") or "unknown")
        if role == "assistant":
            prefix = "<assistant>\n"
            body = assistant_body(message)
            suffix = "\n</assistant>\n"
            pieces.extend((prefix, body, suffix))
            start = length + len(prefix)
            end = start + len(body)
            spans.append((start, end, bool(message.get("tool_calls")) or "thought:" in body.lower() or "<think>" in body.lower()))
            length += len(prefix) + len(body) + len(suffix)
            continue
        if role == "tool":
            name = str(message.get("name") or "")
            call_id = str(message.get("tool_call_id") or "")
            prefix = f'<tool name={canonical_json(name)} call_id={canonical_json(call_id)}>\n'
            body = str(message.get("content") or "")
            suffix = "\n</tool>\n"
        else:
            prefix = f"<{role}>\n"
            body = str(message.get("content") or "")
            suffix = f"\n</{role}>\n"
        pieces.extend((prefix, body, suffix))
        length += len(prefix) + len(body) + len(suffix)
    pieces.append("</transcript>\n")
    if generation:
        pieces.append("<assistant>\n")
    return RenderedProtocol(system=system, user="".join(pieces), assistant_spans=spans)


def flatten_for_local(rendered: RenderedProtocol, *, assistant_after_user: bool = False) -> tuple[str, list[tuple[int, int, bool]]]:
    """Use the exact canonical system/user contents in a raw local sequence."""
    prefix = "<system>\n" + rendered.system + "\n</system>\n<user>\n"
    # Mistral needs the generation marker outside the user turn. Other local
    # families retain the established serialization for comparable replay.
    generation_marker = "<assistant>\n"
    user = rendered.user
    assistant_prefix = ""
    if assistant_after_user and user.endswith(generation_marker):
        user = user[:-len(generation_marker)]
        assistant_prefix = generation_marker
    suffix = "\n</user>\n" + assistant_prefix
    return prefix + user + suffix, [
        (start + len(prefix), end + len(prefix), actionable)
        for start, end, actionable in rendered.assistant_spans
    ]

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def argument_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def values_match(generated: Any, expected: Any) -> bool:
    if isinstance(expected, (dict, list)):
        return generated == expected
    return str(generated).strip() == str(expected).strip()


def normalise_local_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Original local-replay argument normalisation; no prompt is added."""
    output: list[dict[str, Any]] = []
    for message in messages:
        item = json.loads(json.dumps(message))
        for call in item.get("tool_calls") or []:
            function = call.get("function", {})
            if isinstance(function.get("arguments"), str):
                try:
                    function["arguments"] = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    function["arguments"] = {"_raw_arguments": function["arguments"]}
        output.append(item)
    return output


def normalise_online_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Original OpenRouter message normalisation; no prompt is added."""
    output: list[dict[str, Any]] = []
    for raw in messages:
        message = {key: value for key, value in raw.items() if key in {"role", "content", "name", "tool_call_id", "tool_calls"}}
        if message.get("role") == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = dict(call.get("function") or {})
                function["arguments"] = json.dumps(argument_dict(function.get("arguments")), ensure_ascii=False)
                calls.append({"id": str(call.get("id", "call_history")), "type": "function", "function": function})
            message["tool_calls"] = calls
            message.setdefault("content", None)
        if message.get("role") == "tool":
            message.pop("name", None)
            message["content"] = str(message.get("content", ""))
        output.append(message)
    return output


def parse_first_tool_call(text: str) -> tuple[str | None, dict[str, Any], str]:
    """Return tool, arguments and ``valid``/``malformed``/``none`` status.

    A closed tag can still carry a readable function name while its argument
    JSON is invalid.  Keep that name for Tool-hit diagnostics, but mark the
    call malformed so it can never become a strict Full-hit.
    """
    valid_name = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
    if not match:
        function_match = re.search(
            r'"(?:name|function_name)"\s*:\s*"([^"\\]+)".*?"arguments"\s*:\s*(\{.*?\})', text, flags=re.DOTALL
        )
        if function_match:
            try:
                name = function_match.group(1)
                if valid_name.fullmatch(name):
                    return name, argument_dict(json.loads(function_match.group(2))), "valid"
            except json.JSONDecodeError:
                pass
        xml_match = re.search(r"<function=([^>\s]+)>\s*(\{.*?\})\s*</function>", text, flags=re.DOTALL)
        if xml_match:
            try:
                name = xml_match.group(1)
                if valid_name.fullmatch(name):
                    return name, argument_dict(json.loads(xml_match.group(2))), "valid"
            except json.JSONDecodeError:
                pass
        return None, {}, "none"
    body = match.group(1).strip()
    try:
        payload = json.loads(body)
        if isinstance(payload, dict) and isinstance(payload.get("name"), str) and valid_name.fullmatch(payload["name"]):
            return payload["name"], argument_dict(payload.get("arguments")), "valid"
    except json.JSONDecodeError:
        pass

    # Mistral sometimes emits an arguments-first call with quote damage, e.g.
    # {"arguments:{stock:ZETA"},"name:add_to_watchlist"}.  Its closing tag,
    # outer braces, and name field make the boundary unambiguous.  Repair only
    # this bounded representation; never infer a call from surrounding prose.
    mistral_match = re.fullmatch(
        r"\{?\s*['\"]?arguments['\"]?\s*:\s*\{(?P<arguments>.*)\}\s*,\s*"
        r"['\"]?name['\"]?\s*:\s*['\"]?(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)['\"]?\s*\}?",
        body,
        flags=re.DOTALL,
    )
    if mistral_match:
        repaired_arguments = parse_loose_arguments(mistral_match.group("arguments"))
        if repaired_arguments is not None:
            return mistral_match.group("name"), repaired_arguments, "repaired"
    colon_name, separator, colon_arguments = body.partition(":")
    if separator and valid_name.fullmatch(colon_name.strip()):
        try:
            payload = json.loads(colon_arguments.strip())
            if isinstance(payload, dict):
                return colon_name.strip(), payload, "valid"
        except json.JSONDecodeError:
            pass
    arguments = {key.strip(): value.strip() for key, value in re.findall(
        r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", body, flags=re.DOTALL
    )}
    header = re.match(r"^([A-Za-z_][A-Za-z0-9_.-]*)(?:\s|<|$)", body)
    if header and arguments:
        return header.group(1), arguments, "valid"
    # Some hosted models emit malformed JSON such as
    # {"arguments:{...}","name:get_symbol_by_name"}.  There may also be a
    # user argument called ``name`` earlier in the string, so the final name
    # field is the function name.
    malformed_names = re.findall(
        r"['\"]?(?:name|function_name)['\"]?\s*:\s*['\"]?([A-Za-z_][A-Za-z0-9_.-]*)",
        body,
    )
    if malformed_names:
        return malformed_names[-1], {}, "malformed"
    return None, {}, "none"


def parse_loose_arguments(text: str) -> dict[str, Any] | None:
    """Conservatively parse the bounded malformed Mistral argument object.

    Bare words are accepted as strings only after their key/value boundary is
    explicit.  Arrays and nested objects must remain valid JSON.  Returning
    ``None`` keeps the call malformed rather than manufacturing parameters.
    """
    cleaned = re.sub(r"([A-Za-z_][A-Za-z0-9_.-]*)['\"]?\s*:", r"\1:", text.strip())
    if not cleaned:
        return {}
    parts: list[str] = []
    start = depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(cleaned):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth < 0:
                return None
        elif char == "," and depth == 0:
            parts.append(cleaned[start:index])
            start = index + 1
    if depth != 0:
        return None
    parts.append(cleaned[start:])
    value: dict[str, Any] = {}
    for part in parts:
        key, separator, raw = part.partition(":")
        key = key.strip().strip("'\"")
        raw = raw.strip().strip("'\"")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key) or not raw:
            return None
        if raw in {"true", "false", "null"}:
            parsed: Any = {"true": True, "false": False, "null": None}[raw]
        elif re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", raw):
            parsed = json.loads(raw)
        elif raw.startswith("[") or raw.startswith("{"):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return None
        else:
            parsed = raw
        value[key] = parsed
    return value


def load_side(probes_path: Path, labels_path: Path, online: bool) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    labels = {str(row["probe_id"]): row for row in read_jsonl(labels_path)}
    items: list[dict[str, Any]] = []
    for probe in read_jsonl(probes_path):
        probe_id = str(probe.get("trace_id") or probe.get("probe_id") or "")
        label = labels.get(probe_id)
        if label is None:
            raise RuntimeError(f"Probe has no matching label: {probe_id!r}")
        schemas = probe.get("tool_schemas")
        if not isinstance(schemas, list) or not schemas:
            raise RuntimeError(f"Probe {probe_id!r} has no recorded tool_schemas; tool replacement is forbidden.")
        messages = probe.get("messages")
        if not isinstance(messages, list):
            raise RuntimeError(f"Probe {probe_id!r} has no message prefix.")
        items.append({
            "probe_id": probe_id, "card_id": str(label.get("card_id") or ""),
            "benchmark": probe.get("benchmark"), "domain": probe.get("domain"),
            "messages": messages, "tool_schemas": schemas,
            "expected_tool": label.get("expected_tool"),
            "expected_arguments": label.get("expected_arguments") or {},
        })
    return items, labels


def score(item: dict[str, Any], output_tool: str | None, output_arguments: dict[str, Any], completion: str,
          error: str | None = None, call_parse_status: str = "valid",
          response_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    tool_hit = output_tool == item["expected_tool"]
    full_hit = call_parse_status in {"valid", "repaired"} and tool_hit and all(key in output_arguments and values_match(output_arguments[key], value)
                                 for key, value in item["expected_arguments"].items())
    return {**{key: item[key] for key in ("probe_id", "card_id", "benchmark", "domain", "expected_tool", "expected_arguments")},
            "output_tool": output_tool, "output_arguments": output_arguments,
            "tool_hit": tool_hit, "full_hit": full_hit, "call_parse_status": call_parse_status,
            "completion": completion, "error": error,
            **(response_metadata or {})}


def load_local_model(args: argparse.Namespace) -> tuple[Any, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = args.tokenizer_source or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.padding_side, tokenizer.truncation_side = "left", "left"
    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "trust_remote_code": True, "low_cpu_mem_usage": True}
    # ``device_map=balanced`` lets large local models span the visible GPUs.
    # In that mode Accelerate owns placement and a blanket ``model.to('cuda')``
    # would undo the dispatch (or fail when one card is already occupied).
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
        if args.max_memory_per_gpu:
            model_kwargs["max_memory"] = {
                index: f"{args.max_memory_per_gpu}GiB" for index in range(torch.cuda.device_count())
            }
            model_kwargs["max_memory"]["cpu"] = f"{args.cpu_max_memory}GiB"
    # Loading the checkpoint begins on CPU and this Transformers release
    # validates FA2 during construction, before the existing ``model.to``
    # below moves weights onto the selected CUDA device.  SDPA is valid at
    # construction time and dispatches PyTorch's Flash kernel after that move.
    # An explicit CLI choice still takes precedence for reproduction runs.
    model_kwargs["attn_implementation"] = args.attention_implementation or "sdpa"
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    if config.__class__.__name__ == "Mistral3Config":
        from transformers.models.mistral3 import Mistral3ForConditionalGeneration
        model = Mistral3ForConditionalGeneration.from_pretrained(args.base_model, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.adapter:
        # Adapter tensors may be centrally archived while the lightweight
        # PEFT config remains beside the experiment's logs and metadata.
        adapter_config_dir = getattr(args, "adapter_config_dir", None) or args.adapter
        model = get_peft_model(model, LoraConfig.from_pretrained(str(adapter_config_dir)))
        state = load_file(str(args.adapter / "adapter_model.safetensors"))
        state = {key.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"): value
                 for key, value in state.items()}
        # A sharded ``from_pretrained`` leaves modules on the meta device until
        # Accelerate's dispatch hook materializes them.  Plain state-dict copy
        # is a no-op for those parameters; ``assign=True`` installs the LoRA
        # tensors so they participate in the dispatched forward/merge.
        try:
            missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        except TypeError:  # Older torch without the assign keyword.
            missing, unexpected = model.load_state_dict(state, strict=False)
        if args.device_map:
            # ``assign=True`` preserves the checkpoint tensors' CPU device.
            # Align each LoRA pair with its dispatched base linear layer before
            # generation (and before an optional in-memory merge).
            for layer in model.modules():
                base_layer = getattr(layer, "base_layer", None)
                if base_layer is None:
                    continue
                try:
                    target_device = next(base_layer.parameters()).device
                except StopIteration:
                    continue
                if target_device.type == "meta":
                    continue
                for container_name in ("lora_A", "lora_B"):
                    container = getattr(layer, container_name, None)
                    if container is not None:
                        for adapter_layer in container.values():
                            adapter_layer.to(target_device)
        if [key for key in missing if "lora_" in key] or unexpected:
            raise RuntimeError(f"adapter mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}")
    if not args.device_map:
        model.to("cuda")
    model.eval()
    if args.merge_adapter:
        if not args.adapter:
            raise RuntimeError("--merge-adapter requires --adapter")
        model = model.merge_and_unload(safe_merge=False)
    return tokenizer, model


def replay_local(items: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace, tokenizer: Any, model: Any) -> list[dict[str, Any]]:
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    existing = {str(row["probe_id"]): row for row in read_jsonl(results_path)} if results_path.exists() else {}
    pending = [item for item in items if item["probe_id"] not in existing]
    results = dict(existing)
    mistral_boundary = args.model_family.lower() == "mistral" or "mistral" in str(args.base_model).lower()
    with results_path.open("w", encoding="utf-8") as handle:
        for row in existing.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        while pending:
            prepared = []
            # Keep the tail intact.  ``batch_count`` may be lower than the
            # requested batch size for long prompts; it must never make the
            # yet-unprepared probes disappear from the replay queue.
            window, tail = pending[:args.batch_size], pending[args.batch_size:]
            for item in window:
                rendered = render_protocol(item["messages"], item["tool_schemas"], generation=True)
                if mistral_boundary:
                    # Mistral needs a native assistant boundary outside the
                    # outer user turn.
                    prompt, _ = flatten_for_local(rendered, assistant_after_user=True)
                else:
                    # SFT predicts each in-transcript assistant body directly
                    # after this marker while the canonical outer user turn is
                    # still open.  Closing ``</user>`` here changes the prefix
                    # and makes GLM/Qwen continue roles or observations rather
                    # than emit the next action.
                    prompt = "<system>\n" + rendered.system + "\n</system>\n<user>\n" + rendered.user
                input_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
                if hasattr(input_ids, "tolist"):
                    input_ids = input_ids.tolist()
                if input_ids and isinstance(input_ids[0], list):
                    input_ids = input_ids[0]
                input_ids = input_ids[-args.max_input_tokens:]
                prepared.append((item, input_ids, len(input_ids)))
            longest = max(row[2] for row in prepared)
            batch_count = max(1, min(args.batch_size, args.max_batch_tokens // max(1, longest)))
            selected = prepared[:batch_count]
            pending = [row[0] for row in prepared[batch_count:]] + tail
            batch, sequences = [row[0] for row in selected], [row[1] for row in selected]
            max_length = max(len(sequence) for sequence in sequences)
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            inputs = {
                "input_ids": torch.tensor([[pad] * (max_length - len(sequence)) + sequence for sequence in sequences], dtype=torch.long, device=model.device),
                "attention_mask": torch.tensor([[0] * (max_length - len(sequence)) + [1] * len(sequence) for sequence in sequences], dtype=torch.long, device=model.device),
            }
            torch.cuda.empty_cache()
            stop_ids = {tokenizer.eos_token_id}
            for marker in ("</tool_call>", "<|user|>", "<|assistant|>", "<|observation|>", "<|system|>"):
                marker_ids = tokenizer.encode(marker, add_special_tokens=False)
                if len(marker_ids) == 1:
                    stop_ids.add(marker_ids[0])
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                                           pad_token_id=tokenizer.eos_token_id, eos_token_id=sorted(stop_ids))
            width = inputs["input_ids"].shape[1]
            for item, output in zip(batch, generated):
                completion = tokenizer.decode(output[width:], skip_special_tokens=False)
                tool, arguments, parse_status = parse_first_tool_call(completion)
                row = score(item, tool, arguments, completion, call_parse_status=parse_status)
                results[item["probe_id"]] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                _write_replay_progress(out_dir, results, len(items), "local")
    _write_replay_progress(out_dir, results, len(items), "local", complete=True)
    return [results[item["probe_id"]] for item in items]


def resolve_key(key_file: Path | None) -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    if key_file and key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    raise RuntimeError("OPENROUTER_API_KEY or --key-file is required")


def parse_structured_tool_call(message: dict[str, Any]) -> tuple[str | None, dict[str, Any], str] | None:
    """Parse an OpenAI-compatible native tool call when a provider returns one.

    Online probes intentionally score the first tool invocation. Some providers
    serialize it in assistant text, while others return the same invocation in
    ``message.tool_calls`` with an empty ``message.content``. Treating the
    latter as an empty completion needlessly retries a valid response.
    """
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    first = tool_calls[0]
    if not isinstance(first, dict):
        return None
    function = first.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    raw_arguments = function.get("arguments", "{}")
    if not isinstance(name, str) or not name:
        return None
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return name, {}, "malformed"
    else:
        return name, {}, "malformed"
    return name, arguments if isinstance(arguments, dict) else {}, "valid" if isinstance(arguments, dict) else "malformed"


def online_request(args: argparse.Namespace, api_key: str, item: dict[str, Any]) -> tuple[str | None, dict[str, Any], str, str, str | None, dict[str, Any]]:
    rendered = render_protocol(item["messages"], item["tool_schemas"], generation=True)
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": rendered.system},
            {"role": "user", "content": rendered.user},
        ],
        "temperature": 0,
        "max_tokens": args.max_new_tokens,
    }
    if args.disable_reasoning:
        if args.thinking_protocol == "modelark":
            payload["thinking"] = {"type": "disabled"}
        else:
            payload["reasoning"] = {"enabled": False}
    if args.only_provider:
        payload["provider"] = {
            "only": list(args.only_provider),
            "allow_fallbacks": False,
        }
    elif args.ignore_provider:
        payload["provider"] = {"ignore": list(args.ignore_provider)}
    request = urllib.request.Request(args.endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return None, {}, "", "none", f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:800]}", {}
    except Exception as exc:
        return None, {}, "", "none", f"{type(exc).__name__}: {exc}", {}
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    completion = str(message.get("content") or "")
    finish_reason = str(choice.get("finish_reason") or "")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    metadata = {
        "finish_reason": finish_reason or None,
        "provider": str(data.get("provider") or "unknown"),
        "completion_tokens": usage.get("completion_tokens"),
    }
    # Some OpenRouter providers return HTTP 200 with a partial completion and
    # ``finish_reason=error``.  Treat that as a failed request so replay_online
    # retries it instead of scoring the chopped prefix as a model miss.
    if finish_reason == "error":
        provider = str(data.get("provider") or "unknown")
        return None, {}, completion, "none", f"provider finish_reason=error provider={provider}", metadata
    if isinstance(message, dict):
        structured = parse_structured_tool_call(message)
        if structured is not None:
            tool, arguments, parse_status = structured
            if not completion:
                completion = json.dumps({"tool_calls": message.get("tool_calls")}, ensure_ascii=False)
            return tool, arguments, completion, parse_status, None, metadata
    if not completion.strip():
        provider = str(data.get("provider") or "unknown")
        return None, {}, completion, "none", f"empty completion provider={provider} finish_reason={finish_reason or 'unknown'}", metadata
    tool, arguments, parse_status = parse_first_tool_call(completion)
    return tool, arguments, completion, parse_status, None, metadata


def replay_online(items: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    api_key = resolve_key(args.key_file)
    results_path = out_dir / "results.jsonl"
    # A successful row is an immutable checkpoint.  In particular, never
    # rewrite this file at the beginning of a resumed online run: a SIGTERM
    # while the old implementation was rebuilding it could discard already
    # completed probes.  Failed attempts may remain in the append-only log;
    # a later successful row for the same probe supersedes them in ``existing``.
    existing = {str(row["probe_id"]): row for row in read_jsonl(results_path) if not row.get("error")} if results_path.exists() else {}
    pending = [item for item in items if item["probe_id"] not in existing]
    results = dict(existing)

    def one(item: dict[str, Any]) -> dict[str, Any]:
        tool: str | None = None
        arguments: dict[str, Any] = {}
        parse_status = "none"
        completion, error, metadata = "", None, {}
        for attempt in range(args.max_retries + 1):
            tool, arguments, completion, parse_status, error, metadata = online_request(args, api_key, item)
            if error is None:
                break
            time.sleep(min(30, 2 ** attempt))
        return score(item, tool, arguments, completion, error, parse_status, metadata)

    # Append each finished request and flush it before progress is updated.
    # Thus an interruption loses at most in-flight requests, which are safely
    # retried on the next launch; it can never remove a completed checkpoint.
    with results_path.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in concurrent.futures.as_completed([pool.submit(one, item) for item in pending]):
                row = future.result()
                results[row["probe_id"]] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                _write_replay_progress(out_dir, results, len(items), "online")
    _write_replay_progress(out_dir, results, len(items), "online", complete=True)
    return [results[item["probe_id"]] for item in items]


def _write_replay_progress(out_dir: Path, results: dict[str, dict[str, Any]], total: int, mode: str, complete: bool = False) -> None:
    done = len(results)
    value = {"status": "complete" if complete else "running", "mode": mode, "completed": done, "total": total,
             "tool_hits": sum(bool(row.get("tool_hit")) for row in results.values()),
             "full_hits": sum(bool(row.get("full_hit")) for row in results.values())}
    if done:
        value["tool_hit_rate"] = value["tool_hits"] / done
        value["full_hit_rate"] = value["full_hits"] / done
    write_json(out_dir / "progress.json", value)


def upper_tail(wins: int, non_ties: int) -> float:
    return sum(math.comb(non_ties, count) for count in range(wins, non_ties + 1)) / (2**non_ties) if non_ties else 1.0


def paired_card_statistics(real_rows: list[dict[str, Any]], fake_rows: list[dict[str, Any]]) -> dict[str, Any]:
    real = {str(row["probe_id"]): row for row in real_rows}
    fake = {str(row["probe_id"]): row for row in fake_rows}
    card_to_ids: dict[str, list[str]] = defaultdict(list)
    for probe_id, row in real.items():
        if probe_id not in fake:
            continue
        card_to_ids[str(row.get("card_id") or "")].append(probe_id)
    cards = []
    for card_id, probe_ids in sorted(card_to_ids.items()):
        real_full = sum(bool(real[probe_id].get("full_hit")) for probe_id in probe_ids)
        fake_full = sum(bool(fake[probe_id].get("full_hit")) for probe_id in probe_ids)
        difference = real_full - fake_full
        cards.append({"card_id": card_id, "probes": len(probe_ids), "real_full_hits": real_full, "fake_full_hits": fake_full,
                      "real_full_rate": real_full / len(probe_ids), "fake_full_rate": fake_full / len(probe_ids),
                      "outcome": "real_win" if difference > 0 else "fake_win" if difference < 0 else "tie"})
    real_wins = sum(card["outcome"] == "real_win" for card in cards)
    fake_wins = sum(card["outcome"] == "fake_win" for card in cards)
    probes = sum(card["probes"] for card in cards)
    real_full = sum(card["real_full_hits"] for card in cards)
    fake_full = sum(card["fake_full_hits"] for card in cards)
    return {
        "cards": len(cards), "paired_probes": probes,
        "real_card_mean_full_hit": sum(card["real_full_rate"] for card in cards) / len(cards) if cards else 0.0,
        "fake_card_mean_full_hit": sum(card["fake_full_rate"] for card in cards) / len(cards) if cards else 0.0,
        "real_card_any_full_hit_rate": sum(card["real_full_hits"] > 0 for card in cards) / len(cards) if cards else 0.0,
        "fake_card_any_full_hit_rate": sum(card["fake_full_hits"] > 0 for card in cards) / len(cards) if cards else 0.0,
        "real_full_hits": real_full, "fake_full_hits": fake_full,
        "real_full_hit_rate": real_full / probes if probes else 0.0,
        "fake_full_hit_rate": fake_full / probes if probes else 0.0,
        "real_wins": real_wins, "fake_wins": fake_wins, "ties": len(cards) - real_wins - fake_wins,
        "non_tied_cards": real_wins + fake_wins,
        "p_one_sided_exact_sign": upper_tail(real_wins, real_wins + fake_wins),
        "card_outcomes": cards,
    }


def markdown_report(value: dict[str, Any]) -> str:
    return "\n".join([
        "# Paired probe detection results", "",
        "| Metric | Real probes | Fake probes |",
        "|---|---:|---:|",
        f"| Mean full-hit rate per card | {value['real_card_mean_full_hit']:.1%} | {value['fake_card_mean_full_hit']:.1%} |",
        f"| Cards with at least one full hit | {value['real_card_any_full_hit_rate']:.1%} | {value['fake_card_any_full_hit_rate']:.1%} |",
        f"| Overall full-hit rate | {value['real_full_hits']} / {value['paired_probes']} ({value['real_full_hit_rate']:.1%}) | {value['fake_full_hits']} / {value['paired_probes']} ({value['fake_full_hit_rate']:.1%}) |",
        "",
        f"Card outcomes (real wins / fake wins / ties): {value['real_wins']} / {value['fake_wins']} / {value['ties']}. "
        f"Non-tied cards: {value['non_tied_cards']}. One-sided exact sign-test p-value: {value['p_one_sided_exact_sign']:.6g}.",
        "",
        "The p-value uses the sign of (real full hits minus fake full hits) within each release card. "
        "Multiple probes from one card are not treated as independent samples.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="One-file paired probe detector.")
    parser.add_argument("--mode", choices=("local", "online"), required=True)
    parser.add_argument("--bench", choices=BENCHES)
    parser.add_argument("--teacher", choices=TEACHERS)
    parser.add_argument("--condition", default="D_c")
    parser.add_argument("--run-name", help="evaluation model directory name")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--real-probes", type=Path)
    parser.add_argument("--real-labels", type=Path)
    parser.add_argument("--fake-probes", type=Path)
    parser.add_argument("--fake-labels", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--adapter-config-dir", type=Path,
                        help="directory containing adapter_config.json when --adapter stores tensors only")
    parser.add_argument("--tokenizer-source", type=Path)
    parser.add_argument("--model-family", default="glm")
    parser.add_argument("--attention-implementation")
    parser.add_argument("--device-map", default=None,
                        help="Transformers/Accelerate device map (for example, balanced) for multi-GPU local replay")
    parser.add_argument("--max-memory-per-gpu", type=int, default=None,
                        help="GPU memory cap in GiB for Accelerate sharding; permits CPU offload on busy cards")
    parser.add_argument("--cpu-max-memory", type=int, default=64,
                        help="CPU memory cap in GiB used with --max-memory-per-gpu")
    parser.add_argument("--merge-adapter", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--endpoint", default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batch-tokens", type=int, default=18432)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--ignore-provider", action="append", default=[],
                        help="OpenRouter provider slug to skip; may be repeated")
    parser.add_argument("--only-provider", action="append", default=[],
                        help="restrict OpenRouter to these provider slugs and disable provider fallbacks")
    parser.add_argument("--disable-reasoning", action="store_true",
                        help="request non-thinking mode from online models that support it")
    parser.add_argument(
        "--thinking-protocol", choices=("openrouter", "modelark"), default="openrouter",
        help="provider-specific wire format used by --disable-reasoning",
    )
    parser.add_argument("--replay-side", choices=("both", "real", "fake"), default="both",
                        help="replay both probe sides or only one side while preserving the other side")
    parser.add_argument("--report-only", action="store_true",
                        help="write paired statistics from existing result checkpoints without model requests")
    args = parser.parse_args()
    if args.bench:
        if not args.teacher or not args.run_name:
            parser.error("--bench requires --teacher and --run-name")
        source = area_dir(args.bench, "probes", args.teacher, args.output_root) / "S"
        target = area_dir(args.bench, "evaluation", args.teacher, args.output_root) / args.condition / args.run_name
        args.real_probes = args.real_probes or source / "real_probes" / "probes.jsonl"
        args.real_labels = args.real_labels or source / "real_probes" / "labels.jsonl"
        args.fake_probes = args.fake_probes or source / "fake_probes" / "probes.jsonl"
        args.fake_labels = args.fake_labels or source / "fake_probes" / "labels.jsonl"
        args.out_dir = args.out_dir or target
    for name in ("real_probes", "real_labels", "fake_probes", "fake_labels", "out_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required without --bench")
    if args.mode == "local" and args.base_model is None:
        parser.error("--base-model is required in local mode")
    if args.mode == "online" and not args.model:
        parser.error("--model is required in online mode")
    if args.batch_size < 1 or args.workers < 1:
        parser.error("--batch-size and --workers must be positive")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    real_items, _ = load_side(args.real_probes, args.real_labels, online=args.mode == "online")
    fake_items, _ = load_side(args.fake_probes, args.fake_labels, online=args.mode == "online")
    write_json(out_dir / "run_config.json", {
        "mode": args.mode, "real_probes": str(args.real_probes), "fake_probes": str(args.fake_probes),
        "real_labels": str(args.real_labels), "fake_labels": str(args.fake_labels),
        "semantic_scoring": False, "max_new_tokens": args.max_new_tokens,
        "model": args.model if args.mode == "online" else str(args.base_model),
        "ignored_providers": list(args.ignore_provider),
        "only_providers": list(args.only_provider),
        "reasoning_enabled": not args.disable_reasoning,
        "thinking_protocol": args.thinking_protocol,
        "replay_side": args.replay_side,
        "template_policy": PROTOCOL_NAME,
        "structured_tools_api": False,
    })
    if args.report_only:
        real_rows = read_jsonl(out_dir / "real_probes" / "results.jsonl")
        fake_rows = read_jsonl(out_dir / "fake_probes" / "results.jsonl")
    elif args.mode == "local":
        tokenizer, model = load_local_model(args)
        real_rows = (replay_local(real_items, out_dir / "real_probes", args, tokenizer, model)
                     if args.replay_side in {"both", "real"}
                     else read_jsonl(out_dir / "real_probes" / "results.jsonl"))
        fake_rows = (replay_local(fake_items, out_dir / "fake_probes", args, tokenizer, model)
                     if args.replay_side in {"both", "fake"}
                     else read_jsonl(out_dir / "fake_probes" / "results.jsonl"))
    else:
        real_rows = (replay_online(real_items, out_dir / "real_probes", args)
                     if args.replay_side in {"both", "real"}
                     else read_jsonl(out_dir / "real_probes" / "results.jsonl"))
        fake_rows = (replay_online(fake_items, out_dir / "fake_probes", args)
                     if args.replay_side in {"both", "fake"}
                     else read_jsonl(out_dir / "fake_probes" / "results.jsonl"))
    # A single-side replay is intentionally silent at the aggregate level.
    # This prevents concurrent real/fake workers from racing to overwrite the
    # paired report with partial (or 0/0) statistics.
    if args.report_only or args.replay_side == "both":
        statistics = paired_card_statistics(real_rows, fake_rows)
        write_json(out_dir / "detection_results.json", statistics)
        (out_dir / "detection_results.md").write_text(markdown_report(statistics), encoding="utf-8")
        print(markdown_report(statistics), flush=True)


if __name__ == "__main__":
    main()
