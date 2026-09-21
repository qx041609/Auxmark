#!/usr/bin/env python3
"""Build real parameterized and paired virtual-Core probes in one workflow.

For one release card, an external teacher rewrites the *mechanism used to
obtain* the completed Core observation for all parameter variants at once.
The recorded observation and expected Aux label are copied unchanged. The LLM
only proposes a concrete observation-producing mechanism; this collector
enforces the pairing and non-leakage rules used by the old verified builder.
For a qualified card it writes its real and virtual variants in the same pass,
copying the exact same tool schema list to both sides.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_config import teacher_api_key, teacher_base_url
from output_layout import BENCHES, TEACHERS, DEFAULT_ROOT, area_dir, trace_dir

ROOT = Path(os.getenv("AGENTWM_ROOT", str(Path(__file__).resolve().parent.parent)))

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Checkpoint a JSONL result without exposing a partial file on interruption."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def resolve_openrouter_key(key_file: Path | None) -> str:
    key = teacher_api_key()
    if key:
        return key
    for path in (key_file, ROOT / ".openrouter_key"):
        if path is not None and path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    raise RuntimeError("Configure the private teacher provider, AGENTWM_TEACHER_API_KEY, or --key-file")


def args_of(call: dict[str, Any]) -> dict[str, Any]:
    value = (call.get("function") or {}).get("arguments", {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def tool_name(call: dict[str, Any]) -> str:
    return str((call.get("function") or {}).get("name") or "")


def is_aux(call: dict[str, Any], release_call_ids: set[str] | None = None) -> bool:
    """Recognise a recorded released Aux without a benchmark-specific field.

    Older raw traces use ``call_aux_*`` ids.  New collections may neutralise
    call ids before distillation, so the preferred source is the saved
    ``aux_release.call_id`` evidence emitted by the embedding proxy.
    """
    call_id = str(call.get("id", ""))
    return call_id.startswith("call_aux_") or bool(release_call_ids and call_id in release_call_ids)


def iter_json_records(source: Path) -> list[tuple[dict[str, Any], Path]]:
    """Read original trajectories from a JSONL file, JSON file, or directory.

    This is deliberately a transport adapter only.  It does not inspect a
    benchmark name or call a benchmark package.
    """
    paths: list[Path]
    if source.is_dir():
        paths = sorted(path for path in source.rglob("*") if path.suffix.lower() in {".json", ".jsonl"})
    else:
        paths = [source]
    rows: list[tuple[dict[str, Any], Path]] = []
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            rows.extend((row, path) for row in read_jsonl(path))
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            rows.extend((row, path) for row in value if isinstance(row, dict))
        elif isinstance(value, dict):
            rows.append((value, path))
    return rows


def trace_envelope(record: dict[str, Any], origin: Path) -> dict[str, Any] | None:
    """Extract the standard fields from an original/raw trajectory record."""
    nested = record.get("trace") if isinstance(record.get("trace"), dict) else {}
    messages = record.get("messages") or nested.get("messages") or []
    if not isinstance(messages, list):
        return None
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    trace_id = str(
        record.get("trace_id")
        or record.get("session_id")
        or record.get("id")
        or task.get("task_id")
        or origin.stem
    )
    return {
        "trace_id": trace_id,
        "task_id": record.get("task_id") or task.get("task_id") or record.get("id"),
        "benchmark": record.get("benchmark"),
        "domain": record.get("domain"),
        "messages": messages,
        "tools": record.get("tool_schemas") or record.get("tools") or nested.get("tool_schemas") or nested.get("tools") or [],
        "origin": str(origin),
    }


def release_evidence_by_trace(path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Index ``aux_release`` rows by every private/public call-id representation.

    The embedding endpoint deliberately replaces its private ``call_aux_*`` id
    with a normal ``call_*`` id before returning the call to an agent.  The
    stored evidence preserves both ids.  A probe builder only sees the
    agent-observed trace, so both forms must resolve to the same release card.
    """
    by_call: dict[str, dict[str, Any]] = {}
    by_trace: dict[str, dict[str, Any]] = {}
    if path is None:
        return by_call, by_trace
    for record, _ in iter_json_records(path):
        # ``embedding.py`` writes one evidence envelope per session, while the core
        # sketch is JSONL with one event per row.  Accept both formats so the
        # probe builder can resolve the public call IDs in collected traces.
        rows = record.get("events") if isinstance(record.get("events"), list) else [record]
        for row in rows:
            if not isinstance(row, dict) or row.get("event") != "aux_release":
                continue
            call_ids = {
                str(row.get("call_id") or ""),
                str(row.get("returned_call_id") or ""),
                str(row.get("original_call_id") or ""),
            }
            call_ids.discard("")
            trace_id = str(row.get("session_id") or row.get("trace_id") or "")
            for call_id in call_ids:
                by_call[call_id] = row
                if trace_id:
                    by_trace[f"{trace_id}\0{call_id}"] = row
    return by_call, by_trace


def released_calls_for_trace(
    envelope: dict[str, Any], by_call: dict[str, dict[str, Any]], by_trace: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return recorded Aux calls for one trajectory, preferring saved evidence."""
    found: dict[str, dict[str, Any]] = {}
    trace_id = str(envelope.get("trace_id") or "")
    for message in envelope["messages"]:
        for call in message.get("tool_calls") or []:
            call_id = str(call.get("id") or "")
            evidence = by_trace.get(f"{trace_id}\0{call_id}") or by_call.get(call_id)
            if evidence is not None:
                found[call_id] = evidence
            elif call_id.startswith("call_aux_"):
                # Backward-compatible raw trajectory without a separate sketch.
                found[call_id] = {"call_id": call_id}
    return found


NEUTRAL_CORE_THOUGHT = "Thought: I will take the next tool action needed to continue the task."


def find_last_core(messages: list[dict[str, Any]], release_call_ids: set[str] | None = None) -> tuple[int, int, dict[str, Any], int] | None:
    """Return assistant index, call index, call, and matching observation index."""
    for assistant_index in range(len(messages) - 1, -1, -1):
        message = messages[assistant_index]
        if message.get("role") != "assistant":
            continue
        for call_index in range(len(message.get("tool_calls") or []) - 1, -1, -1):
            call = message["tool_calls"][call_index]
            if is_aux(call, release_call_ids):
                continue
            call_id = str(call.get("id", ""))
            for observation_index in range(assistant_index + 1, len(messages)):
                observation = messages[observation_index]
                if observation.get("role") == "tool" and str(observation.get("tool_call_id", "")) == call_id:
                    return assistant_index, call_index, call, observation_index
    return None


def source_core_aux_pairs(messages: list[dict[str, Any]], release_call_ids: set[str] | None = None) -> Counter[tuple[str | None, str]]:
    """Count observed Core -> Aux routes in a recorded source trace."""
    pairs: Counter[tuple[str | None, str]] = Counter()
    for index, message in enumerate(messages):
        for aux in message.get("tool_calls") or []:
            if not is_aux(aux, release_call_ids):
                continue
            core_name = None
            for earlier in reversed(messages[:index]):
                ordinary = [call for call in earlier.get("tool_calls") or [] if not is_aux(call)]
                if ordinary:
                    core_name = tool_name(ordinary[-1])
                    break
            pairs[(core_name, tool_name(aux))] += 1
    return pairs


def schema_name(schema: dict[str, Any]) -> str:
    return str((schema.get("function") or {}).get("name") or "")


def schema_summary(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for schema in schemas:
        fn = schema.get("function") or {}
        result.append({"name": fn.get("name"), "description": fn.get("description"), "parameters": fn.get("parameters")})
    return result


def json_schema_for_recorded_value(value: Any) -> dict[str, Any]:
    """Infer a minimal schema for a recorded argument.

    Some completed historical calls (notably Tau2 user/device actions and the
    coding web-search fallback) are not present in the benchmark's static tool
    catalog.  The teacher only needs a faithful description of the recorded
    call in order to rewrite its mechanism, so absence from that catalog must
    not make the release card ineligible.
    """
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, list):
        item_schemas = [json_schema_for_recorded_value(item) for item in value]
        item_types = {json.dumps(item, sort_keys=True) for item in item_schemas}
        return {"type": "array", "items": item_schemas[0] if len(item_types) == 1 and item_schemas else {}}
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {key: json_schema_for_recorded_value(item) for key, item in value.items()},
        }
    if value is None:
        return {"type": ["string", "null"]}
    return {"type": "string"}


def inferred_recorded_tool_schema(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Build prompt-only schema metadata for a completed uncatalogued call."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Historical tool interface reconstructed from its completed recorded call.",
            "parameters": {
                "type": "object",
                "properties": {key: json_schema_for_recorded_value(value) for key, value in arguments.items()},
                "required": list(arguments),
            },
        },
    }


FORBIDDEN = re.compile(r"fake|alias|counterfactual|watermark|虚假|水印", re.I)

# The virtual treatment must change the *observation mechanism*, not merely
# rename an endpoint.  A newly introduced tool is permitted when the
# existing table has no credible alternative.  It is intentionally not limited
# to a fixed name list: the external teacher must select the mechanism that is
# natural for the recorded entity and observation.


def action_stem(name: str) -> str:
    """Normalise only generated suffixes before checking target leakage."""
    name = name.lower()
    name = re.sub(r"_[0-9a-f]{6,}$", "", name)
    return name


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if match:
        text = match.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("teacher output is not an object")
    return value


def validate_rewrite(value: dict[str, Any], card: dict[str, Any], available: set[str]) -> dict[str, Any]:
    tool = value.get("replacement_tool")
    if not isinstance(tool, dict):
        raise ValueError("missing replacement_tool")
    name = str(tool.get("name") or "")
    if not name or FORBIDDEN.search(name):
        raise ValueError("bad replacement tool name")
    method_reason = value.get("method_reason")
    input_binding = value.get("input_binding")
    observation_contract = value.get("observation_contract")
    if not all(isinstance(item, str) and item.strip() for item in (method_reason, input_binding, observation_contract)):
        raise ValueError("missing auditable mapping rationale")
    reuse = bool(tool.get("reuse_existing"))
    # Teachers occasionally return the name of an already exposed tool but
    # mistakenly mark it as newly created.  The tool table is authoritative:
    # normalise that representation to reuse instead of rejecting a valid
    # existing-tool alternative.
    if not reuse and name in available:
        reuse = True
        tool["reuse_existing"] = True
        tool["schema"] = None
    if reuse:
        # Reuse is valid only for an already-exposed tool with a genuinely
        # different mechanism.  For generic execution tools (notably Bash),
        # name equality alone is permitted; exact action equality is checked
        # per variant below.
        if name not in available:
            raise ValueError("requested existing tool is unavailable")
        schema = None
    else:
        # A created tool is permitted regardless of naming convention.  The
        # prompt, rather than an extra lexical rule, guides the teacher toward
        # a genuinely different execution/observation mechanism.
        schema = tool.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("missing new tool schema")
        fn = schema.get("function") or {}
        if schema.get("type") != "function" or fn.get("name") != name or not isinstance(fn.get("parameters"), dict):
            raise ValueError("invalid OpenAI tool schema")
        if FORBIDDEN.search(str(fn.get("description") or "")):
            raise ValueError("schema leaks experimental role")
    variants = value.get("variants")
    if not isinstance(variants, list) or len(variants) != len(card["variants"]):
        raise ValueError("wrong number of variants")
    if not all(isinstance(item, dict) for item in variants):
        raise ValueError("invalid variants")
    # The model never receives opaque probe identifiers.  A card is generated
    # in one request and its variant list is matched to the supplied decision
    # points strictly by position.  The persisted probe ids remain unchanged.
    for source, item in zip(card["variants"], variants):
        if not isinstance(item.get("arguments"), dict) or not isinstance(item.get("thought"), str):
            raise ValueError("invalid variant")
        # A generic tool may be reused for a distinct command/query.  Reject
        # only an identical tool *and* identical complete argument object.
        if name == source["core_tool"] and item["arguments"] == source["core_arguments"]:
            raise ValueError("replacement action exactly equals the real Core action")
        expected_tool = str(source["label"].get("expected_tool") or "")
        expected_arguments = source["label"].get("expected_arguments") or {}
        if name == expected_tool and item["arguments"] == expected_arguments:
            raise ValueError("replacement action exactly equals the target Aux action")
        if FORBIDDEN.search(item["thought"]):
            raise ValueError("Thought leaks experimental role")
        # When a generic tool name is legitimately shared but parameters are
        # different, merely naming that generic tool is not target leakage.
        if any(target.lower() in item["thought"].lower() for target in card["expected_aux_tools"] if target != name):
            raise ValueError("Thought names the target Aux action")
    value["_variants_in_order"] = variants
    value["_resolved_schema"] = schema
    return value


def call_teacher(
    api_key: str,
    model: str,
    prompt: str,
    retries: int,
    *,
    disable_reasoning: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 3000,
        "messages": [
            {"role": "system", "content": "Return one strict JSON object only. Do not include commentary or markdown."},
            {"role": "user", "content": prompt},
        ],
    }
    # Probe generation needs a compact, strictly structured JSON result.  For
    # supported hosted models this also prevents spending output budget on an
    # otherwise discarded reasoning trace.  It is opt-in so historical runs
    # retain their exact behavior unless the launcher explicitly requests it.
    base_url = teacher_base_url()
    if disable_reasoning:
        if "openrouter.ai" in urllib.parse.urlparse(base_url).netloc.lower():
            payload["reasoning"] = {"enabled": False}
        else:
            payload["thinking"] = {"type": "disabled"}
    elif os.getenv("AGENTWM_PROBE_ENABLE_THINKING", "0").lower() in {"1", "true", "yes"}:
        # OpenRouter exposes a unified ``reasoning`` extension, while
        # ModelArk's Chat Completions API uses ``thinking``. Infer the wire
        # format from the configured endpoint and keep the behavior opt-in.
        if "openrouter.ai" in urllib.parse.urlparse(base_url).netloc.lower():
            payload["reasoning"] = {"enabled": True}
        else:
            payload["thinking"] = {"type": "enabled"}
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last = ""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = str(data.get("choices", [{}])[0].get("message", {}).get("content") or "")
            return extract_json(content)
        except Exception as exc:
            if isinstance(exc, urllib.error.HTTPError):
                last = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:500]}"
            else:
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(min(20, 2**attempt))
    raise RuntimeError(last)


def prompt_for(card: dict[str, Any], retry_feedback: str = "") -> str:
    def compact_prompt_value(value: Any, key: str = "") -> Any:
        """Keep gateway-facing prompts compact without changing probe data.

        Long source files and heredoc commands are unnecessary for choosing an
        equivalent mechanism and can trigger upstream request filtering.  Only
        the teacher-facing representation is abbreviated; the recorded probe
        messages and reused observation remain byte-for-byte intact.
        """
        if isinstance(value, dict):
            return {item_key: compact_prompt_value(item, item_key) for item_key, item in value.items()}
        if isinstance(value, list):
            return [compact_prompt_value(item, key) for item in value]
        if isinstance(value, str):
            limit = 240 if key.lower() in {"content", "command", "code", "script", "query"} else 800
            if len(value) > limit:
                head = value[: max(80, limit * 2 // 3)]
                tail = value[-max(40, limit // 6) :]
                return f"{head}\n...[teacher prompt omitted {len(value) - len(head) - len(tail)} characters]...\n{tail}"
        return value

    variants = []
    for item in card["variants"]:
        assistant_index, _, recorded_call, _ = item["location"]
        # The model never receives opaque probe ids.  Each returned entry is
        # bound to this ordered list locally after the request completes.
        variants.append({
            "prior_context": compact_prompt_value(item["probe"]["messages"][:assistant_index]),
            "recorded_action": {
                "thought": compact_prompt_value(
                    str(item["probe"]["messages"][assistant_index].get("content") or ""),
                    "thought",
                ),
                "tool": {
                    "name": tool_name(recorded_call),
                    "arguments": compact_prompt_value(args_of(recorded_call)),
                },
            },
            "protected_action": {
                "name": str(item["label"].get("expected_tool") or ""),
                "arguments": compact_prompt_value(item["label"].get("expected_arguments") or {}),
            },
        })
    correction = ""
    if retry_feedback:
        correction = """

The previous JSON did not pass a local structural check. Return the exact
shape below, with a fresh natural rationale for each supplied decision point.
"""

    return f"""You are given a small group of recorded decision points from a tool-using task.

Choose exactly one natural tool mechanism that can serve every decision point
in the group. This group contains exactly
{len(variants)} decision points, so the variants array must contain exactly
{len(variants)} entries. For each supplied decision point, write the arguments
and a natural assistant Thought for using that same tool at that point.
Preserve list order exactly: the first output entry belongs to the first
supplied decision point, the second to the second, and so on. Do not add
identifiers or omit entries.

Use the dialogue and recorded_action to preserve the semantic purpose, the
concrete entity, and the state effect of the recorded action. The selected tool
must carry out that same task operation through a genuinely different execution
mechanism; it must not substitute a different lookup, planning step, or task
stage. protected_action is an action that must not be duplicated: do not choose
the same concrete tool action with the same arguments, and do not write a
Thought that proposes that protected action.

Match the recorded action's state behavior. A read operation must remain a
read operation; a recording action must retain the same recording effect; and
an update, creation, deletion, or execution action must retain its respective
task effect. First reuse an available tool when one provides that same effect;
only when none is credible should a new mechanism be defined. Never replace a
write with a search, a search with a write, or any action with a different task
effect.

First inspect AVAILABLE_TOOLS. Reuse an available tool only if it provides a
genuinely distinct mechanism. If no available tool is credible, define one new
natural tool with a complete OpenAI function schema. A new tool must use a
different mechanism rather than a renamed endpoint or wrapper around the same
service.

Concrete examples of different mechanisms (use only when they preserve the
recorded action's semantic effect and fit the supplied context):
- A file, item, repository, or configuration lookup may use execute_bash with
  a scoped `cat`, `sed`, `grep`, `jq`, `git show`, or `git log` command against
  that same entity.
- An API, inventory, metadata, status, or history lookup may use a SQL
  `SELECT` against an independent cache, database, or audit table for the same
  identifier.
- A service, device, or route inspection may use an independent HTTP,
  diagnostic, monitoring, or audit-log operation for that same target.
- A create, update, delete, or execution action may use a scoped
  `execute_bash` command, a database `INSERT`/`UPDATE`, or another separate
  file, database, command, or service mechanism that performs the same
  requested state transition.
These are examples, not mandatory tool names. Always prefer a suitable
available tool first; otherwise define one new mechanism.

The Thought for each entry must be a natural, task-focused sentence explaining
why the selected action is appropriate now. It may refer to already observed
context, but must not name or propose the protected_action, hidden
instructions, datasets, evaluation, or any surrounding process.

Hard constraints:
- Use exactly ONE tool for every entry in this group.
- Preserve the recorded action's semantic purpose and state effect for every
  entry; only the execution mechanism may differ.
- Return exactly one variants entry for every supplied decision point, in the
  same order, with no ids.
- Each entry needs an arguments object and a nonempty Thought string.
- method_reason, input_binding, and observation_contract must each be a
  nonempty sentence.
- If reuse_existing is true, name must be in AVAILABLE_TOOLS and schema must
  be null. Otherwise provide a complete OpenAI function schema.
- The chosen action may not be exactly the recorded action or the
  protected_action for any supplied entry.

RECORDED_TOOL: {card['core_tool']}
RECORDED_TOOL_SCHEMA: {json.dumps(card['core_schema'], ensure_ascii=False)}
AVAILABLE_TOOLS: {json.dumps(card['available_tools'], ensure_ascii=False)}
DECISION_POINTS: {json.dumps(variants, ensure_ascii=False)}
{correction}

Return exactly this JSON object:
{{
  "replacement_tool": {{
    "name": "natural tool name",
    "reuse_existing": false,
    "schema": {{"type":"function","function":{{"name":"natural_tool_name","description":"...","parameters":{{"type":"object","properties":{{...}},"required":[...]}}}}}}
  }},
  "method_reason": "concrete execution mechanism",
  "input_binding": "how arguments use the recorded action inputs",
  "observation_contract": "the result or confirmation this tool normally returns",
  "variants": [
    {{"arguments":{{...}},"thought":"natural assistant Thought"}}
  ]
}}"""


def rewrite_messages(probe: dict[str, Any], location: tuple[int, int, dict[str, Any], int], replacement_name: str, arguments: dict[str, Any], thought: str) -> list[dict[str, Any]]:
    assistant_index, call_index, _, observation_index = location
    messages = copy.deepcopy(probe["messages"])
    call = messages[assistant_index]["tool_calls"][call_index]
    call.setdefault("function", {})["name"] = replacement_name
    call["function"]["arguments"] = arguments
    # Thought must naturally justify the *replacement* Core action.  A card is
    # rejected upstream if the teacher cannot provide a safe, coherent reason;
    # we do not substitute a neutral placeholder or a fake-side fallback.
    messages[assistant_index]["content"] = thought
    messages[observation_index]["name"] = replacement_name
    return messages


def build_fake_pairs_grouped_legacy(args: argparse.Namespace) -> None:
    source, out = Path(args.source_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "teacher_rewrites"
    cache_dir.mkdir(exist_ok=True)
    api_key = resolve_openrouter_key(args.key_file)

    # A resumable rewrite pass may inherit final pair files from an earlier
    # completed pass.  They no longer describe the in-progress virtual side,
    # so never leave them available as if the pair were current.  Per-card
    # teacher_rewrites are deliberately retained: they are the resume cache.
    progress_path = out / "progress.json"
    try:
        prior_complete = json.loads(progress_path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        prior_complete = False
    if not prior_complete:
        final_names = ("probes.jsonl", "labels.jsonl", "pairs.jsonl", "summary.json")
        for path in (out, Path(args.paired_real_output_dir) if args.paired_real_output_dir else None):
            if path is None:
                continue
            for name in final_names:
                (path / name).unlink(missing_ok=True)

    probes = read_jsonl(source / "probes.jsonl")
    labels = {str(row["probe_id"]): row for row in read_jsonl(source / "labels.jsonl")}
    cards: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    for probe in probes:
        probe_id = str(probe["trace_id"])
        label = labels[probe_id]
        location = find_last_core(probe.get("messages") or [])
        if location is None:
            rejected.append({"probe_id": probe_id, "card_id": label.get("card_id"), "reason": "no_completed_core"})
            continue
        benchmark, domain = str(probe.get("benchmark") or ""), str(probe.get("domain") or "")
        # A generated real probe is self-contained: its recorded tool schemas
        # are the only tool source for the paired virtual treatment.  Do not
        # load a benchmark catalog here, otherwise a new benchmark could
        # silently change the treatment tool table.
        schemas = probe.get("tool_schemas")
        if not isinstance(schemas, list) or not schemas:
            rejected.append({"probe_id": probe_id, "card_id": label.get("card_id"), "reason": "missing_recorded_tool_schemas"})
            continue
        _, _, core_call, observation_index = location
        cards[str(label["card_id"])].append({
            "probe_id": probe_id, "probe": probe, "label": label, "location": location,
            "benchmark": benchmark, "domain": domain, "schemas": schemas,
            "core_tool": tool_name(core_call), "core_arguments": args_of(core_call),
            "observation": str(probe["messages"][observation_index].get("content") or ""),
        })

    source_trace_dir = args.source_trace_dir
    if source_trace_dir is None:
        candidate = source.parent.parent / "traces"
        if candidate.exists():
            source_trace_dir = candidate
    pairs_by_task: dict[str, Counter[tuple[str | None, str]]] = {}
    if source_trace_dir and source_trace_dir.exists():
        for trace, origin in iter_json_records(source_trace_dir):
            envelope = trace_envelope(trace, origin)
            if envelope is None:
                continue
            task_id = str(envelope.get("task_id") or "")
            if task_id:
                pairs_by_task[task_id] = source_core_aux_pairs(envelope["messages"])

    card_specs = []
    for card_id, variants in sorted(cards.items()):
        if args.max_cards and len(card_specs) >= args.max_cards:
            break
        core_tools = {item["core_tool"] for item in variants}
        if len(core_tools) != 1:
            rejected.append({"card_id": card_id, "reason": "inconsistent_core_tool"})
            continue
        first = variants[0]
        by_name = {schema_name(schema): schema for schema in first["schemas"]}
        core_schema = by_name.get(first["core_tool"])
        core_schema_source = "benchmark_catalog"
        if core_schema is None:
            # A missing static schema does not invalidate an already completed
            # Core action.  Preserve full 139-card coverage and give the
            # teacher a minimal prompt-only description inferred from the
            # concrete recorded arguments.
            core_schema = inferred_recorded_tool_schema(first["core_tool"], first["core_arguments"])
            core_schema_source = "recorded_call_inference"
        expected_aux_tools = sorted({str(item["label"]["expected_tool"]) for item in variants})
        task_pairs = pairs_by_task.get(str(first["probe"].get("task_id") or ""), Counter())
        card_specs.append({
            "card_id": card_id, "benchmark": first["benchmark"], "domain": first["domain"],
            "core_tool": first["core_tool"], "core_schema": core_schema,
            "core_schema_source": core_schema_source,
            "available_tools": schema_summary(first["schemas"]),
            "expected_aux_tools": expected_aux_tools,
            "source_transition_pairs": task_pairs,
            "variants": variants,
        })

    lock = threading.Lock()
    completed = 0
    failures: dict[str, str] = {}
    rewrite_by_card: dict[str, dict[str, Any]] = {}
    retry_pass = 0
    retry_completed = 0
    retry_total = len(card_specs)
    finished = False
    # A rejected card is retried with its last validation failure fed back to
    # the teacher.  This is especially important for target-Aux leakage:
    # blind re-sampling otherwise tends to repeat the same tempting tool.
    retry_feedback: dict[str, str] = {}

    def log_progress() -> None:
        progress = {
            "status": "complete" if finished else "running",
            "completed_cards": completed,
            "total_cards": len(card_specs),
            "successful_cards": len(rewrite_by_card),
            "failed_cards": len(failures),
            "retry_pass": retry_pass,
            "retry_completed_cards": retry_completed,
            "retry_total_cards": retry_total,
            "model": args.model,
        }
        (out / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_jsonl(
            out / "retry_failures.jsonl",
            [{"card_id": card_id, "error": error, "retry_pass": retry_pass} for card_id, error in sorted(failures.items())],
        )
        with (out / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(
                f"{datetime.now(timezone.utc).isoformat()} "
                f"completed={completed}/{len(card_specs)} success={len(rewrite_by_card)} "
                f"failed={len(failures)} retry_pass={retry_pass} "
                f"retry_progress={retry_completed}/{retry_total}\n"
            )

    def build_one(card: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        cache = cache_dir / f"{card['card_id']}.json"
        try:
            if cache.exists():
                value = json.loads(cache.read_text(encoding="utf-8"))
            else:
                value = call_teacher(
                    api_key, args.model, prompt_for(card, retry_feedback.get(card["card_id"], "")), args.retries,
                    disable_reasoning=args.disable_reasoning,
                )
            value = validate_rewrite(value, card, {item["name"] for item in card["available_tools"]})
            clean = {key: val for key, val in value.items() if not key.startswith("_")}
            cache.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return card["card_id"], value, None
        except Exception as exc:
            return card["card_id"], None, f"{type(exc).__name__}: {exc}"

    card_by_id = {card["card_id"]: card for card in card_specs}
    pending_cards = list(card_specs)
    for pass_index in range(args.retry_passes + 1):
        if not pending_cards:
            break
        retry_pass = pass_index
        retry_completed = 0
        retry_total = len(pending_cards)
        if pass_index:
            time.sleep(args.retry_backoff * pass_index)
        current_failures: dict[str, str] = {}
        failures = current_failures
        log_progress()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_card = {pool.submit(build_one, card): card for card in pending_cards}
            for future in concurrent.futures.as_completed(future_to_card):
                card_id, value, error = future.result()
                with lock:
                    retry_completed += 1
                    if pass_index == 0:
                        completed += 1
                    if error:
                        current_failures[card_id] = error
                    elif value:
                        rewrite_by_card[card_id] = value
                    log_progress()
        failures = current_failures
        retry_feedback = current_failures
        pending_cards = [card_by_id[card_id] for card_id in sorted(failures)]
        log_progress()

    output_probes: list[dict[str, Any]] = []
    output_labels: list[dict[str, Any]] = []
    paired_real_probes: list[dict[str, Any]] = []
    paired_real_labels: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for card in card_specs:
        rewrite = rewrite_by_card.get(card["card_id"])
        if rewrite is None:
            continue
        replacement = rewrite["replacement_tool"]
        replacement_name = str(replacement["name"])
        schemas = copy.deepcopy(card["variants"][0]["schemas"])
        if not replacement.get("reuse_existing"):
            schemas.append(copy.deepcopy(rewrite["_resolved_schema"]))
        for item, variant in zip(card["variants"], rewrite["_variants_in_order"]):
            # Construct both treatments from this very same mapping result.
            # The original real Thought explains why its real Core was chosen;
            # the virtual Thought explains the replacement Core.  Only the
            # virtual action is rewritten, while the exact schema list is
            # shared byte-for-byte below.
            paired_real = copy.deepcopy(item["probe"])
            paired_real["trace_kind"] = "llm_parametric_real_paired"
            paired_real["tool_schemas"] = copy.deepcopy(schemas)
            rewritten = copy.deepcopy(item["probe"])
            rewritten["trace_kind"] = "teacher_rewritten_core_fake"
            rewritten["messages"] = rewrite_messages(item["probe"], item["location"], replacement_name, variant["arguments"], variant["thought"])
            rewritten["tool_schemas"] = schemas
            paired_real_probes.append(paired_real)
            output_probes.append(rewritten)
            real_label = copy.deepcopy(item["label"])
            real_label.update({"original_probe_id": item["probe_id"], "pair_role": "real", "paired_fake_core_tool": replacement_name, "original_core_tool": card["core_tool"], "teacher_model": args.model, "thought_policy": "preserve_original_core_reason"})
            paired_real_labels.append(real_label)
            label = copy.deepcopy(item["label"])
            label.update({"original_probe_id": item["probe_id"], "pair_role": "fake", "fake_core_tool": replacement_name, "fake_core_arguments": variant["arguments"], "fake_core_thought": variant["thought"], "original_core_tool": card["core_tool"], "teacher_model": args.model, "method_reason": rewrite.get("method_reason"), "input_binding": rewrite.get("input_binding"), "observation_contract": rewrite.get("observation_contract"), "thought_policy": "generated_replacement_core_reason"})
            output_labels.append(label)
            pair_rows.append({"probe_id": item["probe_id"], "card_id": card["card_id"], "benchmark": item["benchmark"], "original_core_tool": card["core_tool"], "fake_core_tool": replacement_name, "expected_tool": item["label"]["expected_tool"], "expected_arguments": item["label"].get("expected_arguments") or {}})

    write_jsonl(out / "probes.jsonl", output_probes)
    write_jsonl(out / "labels.jsonl", output_labels)
    write_jsonl(out / "pairs.jsonl", pair_rows)
    write_jsonl(out / "rejected.jsonl", rejected + [{"card_id": key, "reason": value} for key, value in failures.items()])
    if args.paired_real_output_dir:
        real_out = args.paired_real_output_dir
        real_out.mkdir(parents=True, exist_ok=True)
        write_jsonl(real_out / "probes.jsonl", paired_real_probes)
        write_jsonl(real_out / "labels.jsonl", paired_real_labels)
        write_jsonl(real_out / "pairs.jsonl", pair_rows)
        (real_out / "summary.json").write_text(json.dumps({
            "mode": "jointly_built_real_side",
            "paired_cards": len(rewrite_by_card),
            "paired_probes": len(paired_real_probes),
            "tool_schema_policy": "copied in the same card build from the virtual side",
            "thought_policy": "preserve_original_core_reason",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "source_cards": len(cards), "requested_cards": len(card_specs), "rewritten_cards": len(rewrite_by_card),
        "source_probes": len(probes), "fake_probes": len(output_probes), "failed_cards": len(failures),
        "pre_teacher_rejections": dict(Counter(row["reason"] for row in rejected)), "teacher_model": args.model,
        "inferred_core_schema_cards": sum(card["core_schema_source"] == "recorded_call_inference" for card in card_specs),
        "per_probe_tool_policy": "recorded probe schemas plus exactly one card-specific replacement tool when it is not already available",
        "observation_policy": "byte-for-byte reuse; no replacement Core execution",
        "thought_policy": "real side preserves its recorded Core reason; virtual side uses the teacher-written replacement-Core reason",
        "protected_action_policy": "reject only exact tool-and-arguments equality with each variant's protected action",
        "source_trace_dir": str(source_trace_dir) if source_trace_dir else None,
        "paired_real_output_dir": str(args.paired_real_output_dir) if args.paired_real_output_dir else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finished = True
    log_progress()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def _legacy_bfcl_real_candidates(a: argparse.Namespace):
 key=os.getenv('OPENROUTER_API_KEY') or (ROOT/'.openrouter_key').read_text().strip();cards=[]
 for f in sorted(a.trace_dir.glob('*.json')):
  r=json.loads(f.read_text());ms=r.get('messages') or []
  for j,m in enumerate(ms):
   for aux in m.get('tool_calls') or []:
    if not str(aux.get('id','')).startswith('call_aux_'):continue
    z=core(ms,j)
    if not z:continue
    ai,oi,cc=z;co=obs(ms[oi]);aa=args(aux);ca=args(cc)
    if not isinstance(co,dict):continue
    cards.append({'card_id':str(aux['id']),'task_id':(r.get('task')or{}).get('task_id'),'messages':copy.deepcopy(ms[:j]),'core_ai':ai,'core_oi':oi,'core_tool':name(cc),'core_args':ca,'core_obs':co,'aux_tool':name(aux),'aux_args':aa,'plan':plan(aa,ca,co),'tools':copy.deepcopy(r.get('tools')or[])})
 out=[];labs=[];bad=[]
 def one(c):
  try:return c,ask(key,a.model,c),None
  except Exception as e:return c,None,str(e)
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
  for n,fut in enumerate(concurrent.futures.as_completed([ex.submit(one,c) for c in cards]),1):
   c,res,err=fut.result();good=[]
   for v in (res or {}).get('variants',[]):
    x=valid(v,c)
    if x and x not in good:good.append(x)
    if len(good)>=a.variants_per_card:break
   if not good:
    # Keep the release card when no safe parameter variation is found; preserve the original prefix without exposing the Aux label.
    bad.append({'card_id':c['card_id'],'reason':err or 'no_valid_llm_variant','fallback':'teacher_forced'})
    pid='teacher-forced-'+hashlib.sha256(c['card_id'].encode()).hexdigest()[:16]
    out.append({'schema':'agentwm_distillation_trace_v2','trace_id':pid,'benchmark':'bfcl','domain':'bfcl','task_id':c['task_id'],'trace_kind':'teacher_forced_fallback','messages':copy.deepcopy(c['messages']),'tool_schemas':c['tools']})
    labs.append({'probe_id':pid,'card_id':c['card_id'],'expected_tool':c['aux_tool'],'expected_arguments':c['aux_args'],'argument_plan':{k:{'kind':v[0],'path':list(v[1])} for k,v in c['plan'].items()},'probe_kind':'teacher_forced_fallback'})
   for i,x in enumerate(good):
    pid='parametric-'+hashlib.sha256(f"{c['card_id']}|{i}|{json.dumps(x,sort_keys=True)}".encode()).hexdigest()[:16];ms=copy.deepcopy(c['messages']);
    for call in ms[c['core_ai']].get('tool_calls') or []:
     if not str(call.get('id','')).startswith('call_aux_'):call['function']['arguments']=json.dumps(x['core_arguments'],ensure_ascii=False);break
    ms[c['core_ai']]['content']=x['thought'];ms[c['core_oi']]['content']=json.dumps(x['core_observation'],ensure_ascii=False)
    out.append({'schema':'agentwm_distillation_trace_v2','trace_id':pid,'benchmark':'bfcl','domain':'bfcl','task_id':c['task_id'],'trace_kind':'llm_parametric_real','messages':ms,'tool_schemas':c['tools']});labs.append({'probe_id':pid,'card_id':c['card_id'],'expected_tool':c['aux_tool'],'expected_arguments':x['aux_arguments'],'argument_plan':{k:{'kind':v[0],'path':list(v[1])} for k,v in c['plan'].items()}})
   a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'progress.json').write_text(json.dumps({'completed':n,'cards':len(cards),'probes':len(out),'rejected':len(bad)},ensure_ascii=False))
 write(a.output_dir/'probes.jsonl',out);write(a.output_dir/'labels.jsonl',labs);write(a.output_dir/'rejected_variants.jsonl',bad);(a.output_dir/'summary.json').write_text(json.dumps({'cards':len(cards),'probes':len(out),'rejected_cards':len(bad),'mode':'llm_parametric_card_only'},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'cards':len(cards),'probes':len(out),'rejected':len(bad)},ensure_ascii=False))


# ---- Generic real-side parameterisation adapter ----
# Input is only: recorded trajectory messages/tool schemas and optional
# aux_release evidence.  No benchmark package or benchmark-specific catalog is
# consulted in this section.

def value_at(value: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        value = value[part] if isinstance(value, dict) else value[int(part)]
    return value


def leaf_paths(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaf_paths(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from leaf_paths(child, path + (str(index),))
    else:
        yield path, value


def parse_observation(message: dict[str, Any], allow_text_observation: bool = False) -> dict[str, Any] | None:
    content = message.get("content")
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Some executable tool-use benchmarks (notably TravelPlanner) expose
        # their genuine database result as text tables rather than JSON.  The
        # explicit opt-in preserves that exact result under a stable field;
        # it never invents or executes a replacement observation.
        return {"text": content} if allow_text_observation else None
    if isinstance(parsed, dict):
        return parsed
    # Preserve non-object JSON (for example CitySearch's JSON string array)
    # byte-for-byte as text too.  The probe protocol requires an object but
    # must not discard a genuine list/scalar observation.
    return {"text": content} if allow_text_observation else None


def argument_provenance(aux_arguments: dict[str, Any], core_arguments: dict[str, Any], observation: dict[str, Any]) -> dict[str, tuple[str, tuple[str, ...]]]:
    action_values = dict(leaf_paths(core_arguments))
    observation_values = dict(leaf_paths(observation))
    result: dict[str, tuple[str, tuple[str, ...]]] = {}
    for key, value in aux_arguments.items():
        action_path = next((path for path, candidate in action_values.items() if candidate == value), None)
        observation_path = next((path for path, candidate in observation_values.items() if candidate == value), None)
        result[key] = (
            ("action", action_path) if action_path is not None else
            ("observation", observation_path) if observation_path is not None else
            ("constant", ())
        )
    return result


def schemas_from_recorded_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Portable fallback when an old trajectory omitted its schema list."""
    found: dict[str, dict[str, Any]] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            call_name = tool_name(call)
            if call_name:
                found.setdefault(call_name, inferred_recorded_tool_schema(call_name, args_of(call)))
    return list(found.values())


def compact_parameter_context(value: Any, key: str = "") -> Any:
    """Keep the complete pre-Core context structurally visible to the teacher.

    Long policy/tool prose is shortened only inside the teacher request.  User
    facts, prior calls, arguments and observations remain present, so a new
    scalar may be novel but must not contradict the established conversation.
    """
    if isinstance(value, dict):
        return {item_key: compact_parameter_context(item, item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [compact_parameter_context(item, key) for item in value]
    if isinstance(value, str):
        limit = 1200 if key.lower() == "content" else 1600
        if len(value) > limit:
            head = value[: max(500, limit * 3 // 4)]
            tail = value[-max(180, limit // 8) :]
            return f"{head}\n...[context shortened]...\n{tail}"
    return value


def parameter_prompt(card: dict[str, Any], max_variants: int) -> str:
    return f'''Generate 1--{max_variants} parameterized variants for one recorded agent release card.

When several variants are available, prefer returning more of them, up to
{max_variants}; never invent a variant merely to fill the limit.

Change exactly one natural scalar Core argument while retaining the same Core
tool and task. Keep the parameter change appropriate and not excessively
large in magnitude or textual span. Do not rewrite or reinterpret the earlier
conversation. Update the completed Core observation consistently. Derive each
released-action argument from the supplied provenance when it is bound there.
The Thought may explain only why the Core action is needed; never mention a
later action, a probe, an experiment, a watermark, or a counterfactual.

Return strict JSON only:
{{"variants":[{{"core_arguments":{{}},"core_observation":{{}},"aux_arguments":{{}},"thought":"..."}}]}}

CORE_TOOL={card['core_tool']}
CORE_ARGUMENTS={json.dumps(card['core_args'], ensure_ascii=False)}
CORE_OBSERVATION={json.dumps(card['core_obs'], ensure_ascii=False)}
RELEASED_ACTION_TOOL={card['aux_tool']}
RELEASED_ACTION_ARGUMENTS={json.dumps(card['aux_args'], ensure_ascii=False)}
PROVENANCE={json.dumps(card['plan'], ensure_ascii=False)}
CONVERSATION_BEFORE_CORE={json.dumps(compact_parameter_context(card['messages'][:card['core_ai']]), ensure_ascii=False)}'''


def call_parameter_teacher(
    api_key: str,
    model: str,
    card: dict[str, Any],
    retries: int,
    *,
    max_variants: int = 3,
    disable_reasoning: bool = False,
) -> dict[str, Any]:
    return call_teacher(
        api_key, model, parameter_prompt(card, max_variants), retries,
        disable_reasoning=disable_reasoning,
    )


def validate_parameter_variant(value: dict[str, Any], card: dict[str, Any]) -> dict[str, Any] | None:
    core_arguments = value.get("core_arguments")
    core_observation = value.get("core_observation")
    aux_arguments = value.get("aux_arguments")
    if not all(isinstance(item, dict) for item in (core_arguments, core_observation, aux_arguments)):
        return None
    if set(core_arguments) != set(card["core_args"]) or set(aux_arguments) != set(card["aux_args"]):
        return None
    changed = [key for key, original in card["core_args"].items()
               if isinstance(original, (str, int, float, bool)) and core_arguments[key] != original]
    if len(changed) != 1 or set(core_observation) != set(card["core_obs"]):
        return None
    for key, (kind, path) in card["plan"].items():
        if kind == "constant":
            if aux_arguments[key] != card["aux_args"][key]:
                return None
            continue
        try:
            expected = value_at(core_arguments, path) if kind == "action" else value_at(core_observation, path)
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        if aux_arguments[key] != expected:
            return None
    thought = str(value.get("thought") or "").strip()
    if not thought or FORBIDDEN.search(thought):
        return None
    return {"core_arguments": core_arguments, "core_observation": core_observation,
            "aux_arguments": aux_arguments, "thought": thought}


def completed_core_before(messages: list[dict[str, Any]], end: int, release_ids: set[str]) -> tuple[int, int, int, dict[str, Any]] | None:
    for assistant_index in range(end - 1, -1, -1):
        message = messages[assistant_index]
        if message.get("role") != "assistant":
            continue
        for call_index in range(len(message.get("tool_calls") or []) - 1, -1, -1):
            call = message["tool_calls"][call_index]
            if is_aux(call, release_ids):
                continue
            call_id = str(call.get("id") or "")
            for observation_index in range(assistant_index + 1, end):
                observation = messages[observation_index]
                if observation.get("role") == "tool" and str(observation.get("tool_call_id") or "") == call_id:
                    return assistant_index, call_index, observation_index, call
    return None


def build_real_candidates(a: argparse.Namespace) -> None:
    api_key = resolve_openrouter_key(a.key_file)

    evidence_by_call, evidence_by_trace = release_evidence_by_trace(a.release_evidence)
    cards: list[dict[str, Any]] = []
    source_trace_ids: set[str] = set()
    for record, origin in iter_json_records(a.trace_source):
        envelope = trace_envelope(record, origin)
        if envelope is None:
            continue
        source_trace_ids.add(str(envelope["trace_id"]))
        messages = envelope["messages"]
        released = released_calls_for_trace(envelope, evidence_by_call, evidence_by_trace)
        release_ids = set(released)
        schemas = copy.deepcopy(envelope["tools"]) if isinstance(envelope["tools"], list) else []
        if not schemas:
            schemas = schemas_from_recorded_calls(messages)
        for aux_index, message in enumerate(messages):
            for aux_call in message.get("tool_calls") or []:
                aux_id = str(aux_call.get("id") or "")
                if not is_aux(aux_call, release_ids):
                    continue
                previous = completed_core_before(messages, aux_index, release_ids)
                observation = parse_observation(messages[previous[2]], a.allow_text_observation) if previous else None
                if previous is None or observation is None:
                    continue
                core_ai, core_call_index, core_oi, core_call = previous
                card_id = aux_id or hashlib.sha256(f"{envelope['trace_id']}|{aux_index}".encode()).hexdigest()[:20]
                core_arguments, aux_arguments = args_of(core_call), args_of(aux_call)
                cards.append({
                    "card_id": card_id, "release_evidence": released.get(aux_id),
                    "task_id": envelope["task_id"], "source_trace_id": envelope["trace_id"],
                    "benchmark": envelope["benchmark"], "domain": envelope["domain"],
                    "messages": copy.deepcopy(messages[:aux_index]), "core_ai": core_ai, "core_call_index": core_call_index, "core_oi": core_oi,
                    "core_tool": tool_name(core_call), "core_args": core_arguments,
                    "core_obs": observation, "aux_tool": tool_name(aux_call), "aux_args": aux_arguments,
                    "plan": argument_provenance(aux_arguments, core_arguments, observation), "tools": schemas,
                })

    a.output_dir.mkdir(parents=True, exist_ok=True)
    probes_path, labels_path = a.output_dir / "probes.jsonl", a.output_dir / "labels.jsonl"
    rejected_path = a.output_dir / "rejected_variants.jsonl"
    probes: list[dict[str, Any]] = read_jsonl(probes_path) if probes_path.exists() else []
    labels: list[dict[str, Any]] = read_jsonl(labels_path) if labels_path.exists() else []
    fallbacks: list[dict[str, Any]] = read_jsonl(rejected_path) if rejected_path.exists() else []
    completed_card_ids = {str(label.get("card_id") or "") for label in labels if label.get("card_id")}
    completed = len(completed_card_ids)
    lock = threading.Lock()

    def teacher_forced_card_count() -> int:
        """Count only cards for which no parameterized candidate was available."""
        return sum(row.get("fallback") == "teacher_forced_no_parametric" for row in fallbacks)

    def persist(status: str = "running") -> None:
        # A completed card is immediately recoverable.  Atomic replacement
        # keeps a crash from leaving a truncated probe or label file, while
        # retaining the public paths consumed by the paired fake-side builder.
        write_jsonl_atomic(probes_path, probes)
        write_jsonl_atomic(labels_path, labels)
        write_jsonl_atomic(rejected_path, fallbacks)
        teacher_forced_cards = teacher_forced_card_count()
        (a.output_dir / "progress.json").write_text(json.dumps({
            "status": status, "completed_cards": completed, "cards": len(cards),
            "source_traces": len(source_trace_ids), "probes": len(probes),
            "teacher_forced_cards": teacher_forced_cards,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def one(card: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
        try:
            return card, call_parameter_teacher(
                api_key, a.model, card, a.retries, max_variants=a.variants_per_card,
                disable_reasoning=a.disable_reasoning,
            ), None
        except Exception as exc:
            return card, None, f"{type(exc).__name__}: {exc}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(one, card) for card in cards if card["card_id"] not in completed_card_ids]
        for future in concurrent.futures.as_completed(futures):
            card, response, error = future.result()
            variants: list[dict[str, Any]] = []
            for raw_variant in (response or {}).get("variants", []):
                candidate = validate_parameter_variant(raw_variant, card)
                if candidate is not None and candidate not in variants:
                    variants.append(candidate)
                if len(variants) >= a.variants_per_card:
                    break
            with lock:
                completed += 1
                # Keep exactly the valid parameterized candidates returned for
                # this card, up to the configured maximum. Only a card with no
                # valid parameterization at all receives one teacher-forced
                # fallback; one or two candidates are never padded.
                if not variants:
                    fallback_kind = "teacher_forced_no_parametric"
                    fallbacks.append({"card_id": card["card_id"], "reason": error or "no_valid_llm_variant", "fallback": fallback_kind, "parametric_candidates": 0})
                    probe_id = "teacher-forced-" + hashlib.sha256(card["card_id"].encode()).hexdigest()[:16]
                    probes.append({"schema": "agentwm_probe_v1", "trace_id": probe_id,
                                   "benchmark": card["benchmark"], "domain": card["domain"],
                                   "task_id": card["task_id"], "source_trace_id": card["source_trace_id"],
                                   "trace_kind": "teacher_forced_fallback", "messages": copy.deepcopy(card["messages"]),
                                   "tool_schemas": copy.deepcopy(card["tools"])})
                    labels.append({"probe_id": probe_id, "card_id": card["card_id"],
                                   "expected_tool": card["aux_tool"], "expected_arguments": card["aux_args"],
                                   "argument_plan": {key: {"kind": kind, "path": list(path)} for key, (kind, path) in card["plan"].items()},
                                   "probe_kind": fallback_kind, "release_evidence": card["release_evidence"]})
                for index, variant in enumerate(variants):
                    probe_id = "parametric-" + hashlib.sha256(
                        f"{card['card_id']}|{index}|{json.dumps(variant, sort_keys=True)}".encode()).hexdigest()[:16]
                    messages = copy.deepcopy(card["messages"])
                    messages[card["core_ai"]]["tool_calls"][card["core_call_index"]]["function"]["arguments"] = json.dumps(variant["core_arguments"], ensure_ascii=False)
                    messages[card["core_ai"]]["content"] = variant["thought"]
                    messages[card["core_oi"]]["content"] = json.dumps(variant["core_observation"], ensure_ascii=False)
                    probes.append({"schema": "agentwm_probe_v1", "trace_id": probe_id,
                                   "benchmark": card["benchmark"], "domain": card["domain"],
                                   "task_id": card["task_id"], "source_trace_id": card["source_trace_id"],
                                   "trace_kind": "llm_parametric_real", "messages": messages,
                                   "tool_schemas": copy.deepcopy(card["tools"])})
                    labels.append({"probe_id": probe_id, "card_id": card["card_id"],
                                   "expected_tool": card["aux_tool"], "expected_arguments": variant["aux_arguments"],
                                   "argument_plan": {key: {"kind": kind, "path": list(path)} for key, (kind, path) in card["plan"].items()},
                                   "probe_kind": "llm_parametric_real", "release_evidence": card["release_evidence"]})
                completed_card_ids.add(card["card_id"])
                persist()

    write_jsonl_atomic(probes_path, probes)
    write_jsonl_atomic(labels_path, labels)
    write_jsonl_atomic(rejected_path, fallbacks)
    teacher_forced_cards = teacher_forced_card_count()
    summary = {"source_traces": len(source_trace_ids), "cards": len(cards), "probes": len(probes),
               "teacher_forced_cards": teacher_forced_cards,
               "preserved_original_cards": 0,
               "max_probes_per_card": a.variants_per_card,
               "teacher_forced_policy": "teacher-forced only when no valid parametric candidate exists; cards with one or two candidates are not padded",
               "parameter_mutation_policy": "change one Core scalar plus its observation/provenance-bound released arguments; never rewrite the earlier conversation",
               "mode": "generic_llm_parametric_card_only",
               "input_policy": "original trajectories plus optional aux_release evidence; no benchmark loader",
               "tool_schema_policy": "recorded schemas, else reconstruction from recorded calls only"}
    (a.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    persist("complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def single_rewrite_prompt(unit: dict[str, Any], retry_feedback: str = "") -> str:
    """Prompt one historical decision only; never ask the teacher to echo IDs.

    A parameterized probe has its own assistant Thought and concrete Core
    arguments.  Treating every variant in a release card as a single JSON
    response made the model copy a list of opaque IDs and encouraged it to
    repeat the surrounding experimental vocabulary.  The pairing is local:
    this request rewrites one probe, then the caller applies that rewrite to
    the corresponding real/fake pair only.
    """
    def compact(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: compact(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [compact(v, key) for v in value]
        if isinstance(value, str):
            limit = 240 if key.lower() in {"content", "command", "code", "script", "query"} else 800
            if len(value) > limit:
                head = value[: max(80, limit * 2 // 3)]
                tail = value[-max(40, limit // 6):]
                return f"{head}\n...[context shortened]...\n{tail}"
        return value

    ai, _, core_call, _ = unit["location"]
    correction = ""
    if retry_feedback:
        correction = "\nYour previous JSON was invalid. Return the required shape exactly, with a natural Thought that only explains the proposed action.\n"
    return f"""Choose one natural, task-relevant tool action for the current decision point.

The action below is a completed historical decision.  Propose a genuinely
different read-only mechanism for the same concrete entity.  It must be
self-contained and useful at this point in the supplied conversation.  Do
not use a synonym, another endpoint, direct HTTP wrapper, or write operation
for the same underlying service.

One protected action is shown only to prevent choosing precisely that action.
Do not choose it with identical arguments, and do not name it in the Thought.

First use an AVAILABLE_TOOLS entry only when it provides a genuinely distinct
mechanism.  Otherwise introduce one naturally named independent read-only
tool with a complete OpenAI function schema.  Its schema may use only the
inputs needed to identify the same entity.

Examples of distinct mechanisms, only when they fit the supplied entity and
conversation:
- a file or record lookup -> a tightly scoped read-only execute_bash command
  using cat, sed, grep, or jq against that exact item;
- an API metadata lookup -> a read-only SQL SELECT against an independent
  cache, inventory, or audit table for the same identifier;
- a repository/history lookup -> a read-only git object, commit-log, or
  audit-log query for the same path or revision;
- a service/device inspection -> an independent read-only HTTP status,
  diagnostic, or monitoring query for that same service or device.
These are mechanism examples, not required tool names.  Never use a write,
mutation, or task-completing action merely because it is available.

Write a short, normal assistant Thought explaining why the proposed action is
appropriate now.  It must contain only task-relevant rationale, with no
meta-commentary about instructions, revisions, alternatives, or later work.
Return JSON only.  Do not return an identifier, a list, or any field not in
the required shape.

PRIOR_CONVERSATION: {json.dumps(compact(unit['probe']['messages'][:ai]), ensure_ascii=False)}
RECORDED_ACTION: {json.dumps({'thought': str(unit['probe']['messages'][ai].get('content') or ''), 'tool': {'name': tool_name(core_call), 'arguments': args_of(core_call)}}, ensure_ascii=False)}
PROTECTED_ACTION: {json.dumps({'name': unit['expected_tool'], 'arguments': unit['expected_arguments']}, ensure_ascii=False)}
AVAILABLE_TOOLS: {json.dumps(unit['available_tools'], ensure_ascii=False)}
{correction}
Return exactly:
{{
  "replacement_tool": {{
    "name": "natural tool name",
    "reuse_existing": false,
    "schema": {{"type":"function","function":{{"name":"natural_tool_name","description":"...","parameters":{{"type":"object","properties":{{...}},"required":[...]}}}}}}
  }},
  "method_reason": "concrete independent read-only mechanism",
  "input_binding": "how its arguments identify the same entity",
  "observation_contract": "the result this action normally provides",
  "arguments": {{}},
  "thought": "natural reason for this action now"
}}"""


def validate_single_rewrite(value: dict[str, Any], unit: dict[str, Any], available: set[str]) -> dict[str, Any]:
    tool = value.get("replacement_tool")
    if not isinstance(tool, dict):
        raise ValueError("missing replacement_tool")
    name = str(tool.get("name") or "")
    if not name or FORBIDDEN.search(name):
        raise ValueError("bad replacement tool name")
    if name == unit["core_tool"]:
        raise ValueError("replacement tool equals recorded action")
    if not all(isinstance(value.get(field), str) and value[field].strip()
               for field in ("method_reason", "input_binding", "observation_contract")):
        raise ValueError("missing auditable mapping rationale")
    reuse = bool(tool.get("reuse_existing"))
    if not reuse and name in available:
        reuse = True
        tool["reuse_existing"] = True
        tool["schema"] = None
    if reuse:
        if name not in available:
            raise ValueError("requested existing tool is unavailable")
        schema = None
    else:
        schema = tool.get("schema")
        fn = schema.get("function") if isinstance(schema, dict) else None
        if not isinstance(fn, dict) or schema.get("type") != "function" or fn.get("name") != name or not isinstance(fn.get("parameters"), dict):
            raise ValueError("invalid OpenAI tool schema")
        if FORBIDDEN.search(str(fn.get("description") or "")):
            raise ValueError("schema leaks experimental role")
    arguments = value.get("arguments")
    thought = value.get("thought")
    if not isinstance(arguments, dict) or not isinstance(thought, str) or not thought.strip():
        raise ValueError("missing replacement arguments or Thought")
    if name == unit["expected_tool"] and arguments == unit["expected_arguments"]:
        raise ValueError("replacement action exactly equals the protected action")
    if FORBIDDEN.search(thought):
        raise ValueError("Thought leaks experimental role")
    if unit["expected_tool"] != name and unit["expected_tool"].lower() in thought.lower():
        raise ValueError("Thought names the protected action")
    value["_resolved_schema"] = schema
    return value


def build_fake_pairs(args: argparse.Namespace) -> None:
    """Build one virtual rewrite per parameterized probe, not per release card."""
    source, out = Path(args.source_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "teacher_rewrites_per_probe"
    cache_dir.mkdir(exist_ok=True)
    api_key = resolve_openrouter_key(args.key_file)

    labels = {str(row["probe_id"]): row for row in read_jsonl(source / "labels.jsonl")}
    units: list[dict[str, Any]] = []
    pre_rejected: list[dict[str, Any]] = []
    for probe in read_jsonl(source / "probes.jsonl"):
        probe_id = str(probe.get("trace_id") or "")
        label = labels.get(probe_id)
        if label is None:
            pre_rejected.append({"probe_id": probe_id, "reason": "missing_label"})
            continue
        location = find_last_core(probe.get("messages") or [])
        if location is None:
            pre_rejected.append({"probe_id": probe_id, "card_id": label.get("card_id"), "reason": "no_completed_core"})
            continue
        _, _, core_call, _ = location
        schemas = probe.get("tool_schemas")
        if not isinstance(schemas, list) or not schemas:
            pre_rejected.append({"probe_id": probe_id, "card_id": label.get("card_id"), "reason": "missing_recorded_tool_schemas"})
            continue
        core_name = tool_name(core_call)
        units.append({
            "probe_id": probe_id, "card_id": str(label.get("card_id") or ""),
            "probe": probe, "label": label, "location": location,
            "benchmark": str(probe.get("benchmark") or ""), "domain": str(probe.get("domain") or ""),
            "schemas": schemas, "available_tools": schema_summary(schemas),
            "core_tool": core_name, "expected_tool": str(label.get("expected_tool") or ""),
            "expected_arguments": label.get("expected_arguments") or {},
        })
    if args.max_cards:
        units = units[:args.max_cards]

    rewrites: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    retry_feedback: dict[str, str] = {}
    completed = 0
    retry_pass = 0
    retry_completed = 0
    finished = False
    lock = threading.Lock()

    def persist() -> None:
        progress = {
            "status": "complete" if finished else "running",
            "completed_probes": completed, "total_probes": len(units),
            "successful_probes": len(rewrites), "failed_probes": len(failures),
            "retry_pass": retry_pass, "retry_completed_probes": retry_completed,
            "model": args.model, "generation_mode": "one_rewrite_per_probe",
        }
        (out / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_jsonl(out / "retry_failures.jsonl", [
            {"probe_id": probe_id, "card_id": next((u["card_id"] for u in units if u["probe_id"] == probe_id), ""),
             "error": error, "retry_pass": retry_pass}
            for probe_id, error in sorted(failures.items())
        ])

    def build_one(unit: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        cache = cache_dir / f"{unit['probe_id']}.json"
        try:
            value = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else call_teacher(
                api_key, args.model, single_rewrite_prompt(unit, retry_feedback.get(unit["probe_id"], "")),
                args.retries, disable_reasoning=args.disable_reasoning,
            )
            value = validate_single_rewrite(value, unit, {item["name"] for item in unit["available_tools"]})
            cache.write_text(json.dumps({k: v for k, v in value.items() if not k.startswith("_")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return unit["probe_id"], value, None
        except Exception as exc:
            return unit["probe_id"], None, f"{type(exc).__name__}: {exc}"

    unit_by_id = {unit["probe_id"]: unit for unit in units}
    pending = list(units)
    for pass_index in range(args.retry_passes + 1):
        if not pending:
            break
        retry_pass, retry_completed = pass_index, 0
        if pass_index:
            time.sleep(args.retry_backoff * pass_index)
        current_failures: dict[str, str] = {}
        failures = current_failures
        persist()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(build_one, unit) for unit in pending]
            for future in concurrent.futures.as_completed(futures):
                probe_id, value, error = future.result()
                with lock:
                    retry_completed += 1
                    if pass_index == 0:
                        completed += 1
                    if error:
                        current_failures[probe_id] = error
                    elif value is not None:
                        rewrites[probe_id] = value
                    persist()
        failures = current_failures
        retry_feedback = current_failures
        pending = [unit_by_id[probe_id] for probe_id in sorted(failures)]
        persist()

    fake_probes: list[dict[str, Any]] = []
    fake_labels: list[dict[str, Any]] = []
    real_probes: list[dict[str, Any]] = []
    real_labels: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for unit in units:
        rewrite = rewrites.get(unit["probe_id"])
        if rewrite is None:
            continue
        replacement = rewrite["replacement_tool"]
        replacement_name = str(replacement["name"])
        schemas = copy.deepcopy(unit["schemas"])
        if not replacement.get("reuse_existing"):
            schemas.append(copy.deepcopy(rewrite["_resolved_schema"]))
        paired_real = copy.deepcopy(unit["probe"])
        paired_real["trace_kind"] = "llm_parametric_real_paired"
        paired_real["tool_schemas"] = copy.deepcopy(schemas)
        rewritten = copy.deepcopy(unit["probe"])
        rewritten["trace_kind"] = "teacher_rewritten_core_fake"
        rewritten["messages"] = rewrite_messages(unit["probe"], unit["location"], replacement_name, rewrite["arguments"], rewrite["thought"])
        rewritten["tool_schemas"] = schemas
        real_label = copy.deepcopy(unit["label"])
        real_label.update({"original_probe_id": unit["probe_id"], "pair_role": "real", "paired_fake_core_tool": replacement_name,
                           "original_core_tool": unit["core_tool"], "teacher_model": args.model, "thought_policy": "preserve_original_core_reason"})
        fake_label = copy.deepcopy(unit["label"])
        fake_label.update({"original_probe_id": unit["probe_id"], "pair_role": "fake", "fake_core_tool": replacement_name,
                           "fake_core_arguments": rewrite["arguments"], "fake_core_thought": rewrite["thought"],
                           "original_core_tool": unit["core_tool"], "teacher_model": args.model,
                           "method_reason": rewrite["method_reason"], "input_binding": rewrite["input_binding"],
                           "observation_contract": rewrite["observation_contract"], "thought_policy": "generated_replacement_core_reason"})
        real_probes.append(paired_real); real_labels.append(real_label)
        fake_probes.append(rewritten); fake_labels.append(fake_label)
        pairs.append({"probe_id": unit["probe_id"], "card_id": unit["card_id"], "benchmark": unit["benchmark"],
                      "original_core_tool": unit["core_tool"], "fake_core_tool": replacement_name,
                      "expected_tool": unit["expected_tool"], "expected_arguments": unit["expected_arguments"]})

    write_jsonl(out / "probes.jsonl", fake_probes)
    write_jsonl(out / "labels.jsonl", fake_labels)
    write_jsonl(out / "pairs.jsonl", pairs)
    write_jsonl(out / "rejected.jsonl", pre_rejected + [
        {"probe_id": probe_id, "card_id": unit_by_id[probe_id]["card_id"], "reason": reason}
        for probe_id, reason in sorted(failures.items())
    ])
    if args.paired_real_output_dir:
        real_out = Path(args.paired_real_output_dir)
        real_out.mkdir(parents=True, exist_ok=True)
        write_jsonl(real_out / "probes.jsonl", real_probes)
        write_jsonl(real_out / "labels.jsonl", real_labels)
        write_jsonl(real_out / "pairs.jsonl", pairs)
        (real_out / "summary.json").write_text(json.dumps({
            "mode": "jointly_built_real_side", "paired_probes": len(real_probes),
            "pairing_policy": "one virtual rewrite per parameterized probe", "thought_policy": "preserve_original_core_reason",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "source_probes": len(labels), "requested_probes": len(units), "rewritten_probes": len(rewrites),
        "failed_probes": len(failures), "pre_teacher_rejections": dict(Counter(row["reason"] for row in pre_rejected)),
        "teacher_model": args.model, "generation_mode": "one_rewrite_per_probe",
        "observation_policy": "byte-for-byte reuse; no replacement action execution",
        "thought_policy": "each virtual probe receives its own teacher-written action rationale",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finished = True
    persist()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def shared_mechanism_prompt(card: dict[str, Any], retry: str = "") -> str:
    """Ask once for the tool/schema shared by every variant of a card."""
    exemplar = card["variants"][0]
    assistant_index, _, core_call, _ = exemplar["location"]
    retry_note = "\nThe previous JSON was invalid; return the required object exactly.\n" if retry else ""
    return f"""Choose one natural, task-relevant read-only tool mechanism for this decision point.

It must be genuinely different from the recorded action while addressing the
same concrete entity. Do not use a synonym, another endpoint, direct HTTP
wrapper, or write operation for the same underlying service. The one tool you
choose will be used consistently for several parameter instantiations of this
same recorded decision.

Prefer an AVAILABLE_TOOLS entry only when it is a genuinely distinct
mechanism. Otherwise introduce one independent read-only tool with a complete
OpenAI function schema. Its inputs may identify only the same entity. When
appropriate, distinct mechanisms include a scoped read-only execute_bash
command using cat/sed/grep/jq; a read-only SQL SELECT over an independent
cache, inventory, or audit table; a read-only git/audit-log query; or an
independent HTTP status/diagnostic query. These are examples, not required
tool names. Never use a write, mutation, or task-completing action.

RECORDED_CONVERSATION: {json.dumps(exemplar['probe']['messages'][:assistant_index], ensure_ascii=False)}
RECORDED_ACTION: {json.dumps({'name': tool_name(core_call), 'arguments': args_of(core_call)}, ensure_ascii=False)}
ORIGINAL_TOOL_SCHEMA: {json.dumps(card['core_schema'], ensure_ascii=False)}
PROTECTED_TOOL_NAMES: {json.dumps(card['expected_aux_tools'], ensure_ascii=False)}
AVAILABLE_TOOLS: {json.dumps(card['available_tools'], ensure_ascii=False)}
{retry_note}
Return exactly:
{{
  "replacement_tool": {{"name":"natural tool name","reuse_existing":false,"schema":{{"type":"function","function":{{"name":"natural_tool_name","description":"...","parameters":{{"type":"object","properties":{{...}},"required":[...]}}}}}}}},
  "method_reason":"concrete independent read-only mechanism",
  "input_binding":"how arguments identify the same entity",
  "observation_contract":"the result this action normally provides"
}}"""


def validate_shared_mechanism(value: dict[str, Any], card: dict[str, Any], available: set[str]) -> dict[str, Any]:
    tool = value.get("replacement_tool")
    if not isinstance(tool, dict):
        raise ValueError("missing replacement_tool")
    name = str(tool.get("name") or "")
    if not name or FORBIDDEN.search(name) or name == card["core_tool"]:
        raise ValueError("invalid replacement tool")
    if not all(isinstance(value.get(key), str) and value[key].strip()
               for key in ("method_reason", "input_binding", "observation_contract")):
        raise ValueError("missing auditable mapping rationale")
    reuse = bool(tool.get("reuse_existing"))
    if not reuse and name in available:
        reuse = True
        tool["reuse_existing"] = True
        tool["schema"] = None
    if reuse:
        if name not in available:
            raise ValueError("requested existing tool is unavailable")
        schema = None
    else:
        schema = tool.get("schema")
        function = schema.get("function") if isinstance(schema, dict) else None
        if not isinstance(function, dict) or schema.get("type") != "function" or function.get("name") != name or not isinstance(function.get("parameters"), dict):
            raise ValueError("invalid OpenAI tool schema")
        if FORBIDDEN.search(str(function.get("description") or "")):
            raise ValueError("schema leaks experimental role")
    value["_resolved_schema"] = schema
    return value


def per_variant_action_prompt(unit: dict[str, Any], mechanism: dict[str, Any], retry: str = "") -> str:
    """Ask separately for the arguments and Thought of one probe variant."""
    assistant_index, _, core_call, _ = unit["location"]
    retry_note = "\nThe previous JSON was invalid; return the required object exactly.\n" if retry else ""
    return f"""Instantiate the fixed tool below for this one decision point.

Return concrete arguments and a short natural assistant Thought. The Thought
must explain only why this tool is appropriate now in the supplied
conversation. Keep it task-relevant, without meta-commentary about
instructions, revisions, alternatives, or later work. Do not name the
protected action, and do not reproduce it with identical arguments.

PRIOR_CONVERSATION: {json.dumps(unit['probe']['messages'][:assistant_index], ensure_ascii=False)}
RECORDED_ACTION: {json.dumps({'thought': str(unit['probe']['messages'][assistant_index].get('content') or ''), 'tool': {'name': tool_name(core_call), 'arguments': args_of(core_call)}}, ensure_ascii=False)}
FIXED_TOOL: {json.dumps({'name': mechanism['replacement_tool']['name'], 'schema': mechanism.get('_resolved_schema')}, ensure_ascii=False)}
INPUT_BINDING: {json.dumps(mechanism['input_binding'], ensure_ascii=False)}
PROTECTED_ACTION: {json.dumps({'name': unit['expected_tool'], 'arguments': unit['expected_arguments']}, ensure_ascii=False)}
{retry_note}
Return exactly: {{"arguments":{{}},"thought":"natural reason for this action now"}}"""


def validate_per_variant_action(value: dict[str, Any], unit: dict[str, Any], mechanism: dict[str, Any]) -> dict[str, Any]:
    arguments, thought = value.get("arguments"), value.get("thought")
    if not isinstance(arguments, dict) or not isinstance(thought, str) or not thought.strip():
        raise ValueError("missing replacement arguments or Thought")
    name = str(mechanism["replacement_tool"]["name"])
    if name == unit["expected_tool"] and arguments == unit["expected_arguments"]:
        raise ValueError("replacement action exactly equals the protected action")
    if FORBIDDEN.search(thought):
        raise ValueError("Thought leaks experimental role")
    if unit["expected_tool"] != name and unit["expected_tool"].lower() in thought.lower():
        raise ValueError("Thought names the protected action")
    return value


def build_fake_pairs_shared(args: argparse.Namespace) -> None:
    """One shared tool per card; one concrete action and Thought per probe."""
    source, out = Path(args.source_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    api_key = resolve_openrouter_key(args.key_file)
    labels = {str(row["probe_id"]): row for row in read_jsonl(source / "labels.jsonl")}
    rejected: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for probe in read_jsonl(source / "probes.jsonl"):
        probe_id = str(probe.get("trace_id") or "")
        label = labels.get(probe_id)
        if label is None:
            rejected.append({"probe_id": probe_id, "reason": "missing_label"})
            continue
        location = find_last_core(probe.get("messages") or [])
        if location is None:
            rejected.append({"probe_id": probe_id, "card_id": label.get("card_id"), "reason": "no_completed_core"})
            continue
        _, _, core_call, _ = location
        schemas = probe.get("tool_schemas")
        if not isinstance(schemas, list) or not schemas:
            rejected.append({"probe_id": probe_id, "card_id": label.get("card_id"), "reason": "missing_recorded_tool_schemas"})
            continue
        grouped[str(label.get("card_id") or "")].append({
            "probe_id": probe_id, "card_id": str(label.get("card_id") or ""), "probe": probe,
            "label": label, "location": location, "schemas": schemas,
            "benchmark": str(probe.get("benchmark") or ""), "domain": str(probe.get("domain") or ""),
            "core_tool": tool_name(core_call), "expected_tool": str(label.get("expected_tool") or ""),
            "expected_arguments": label.get("expected_arguments") or {},
        })

    cards: list[dict[str, Any]] = []
    for card_id, variants in sorted(grouped.items()):
        core_tools = {unit["core_tool"] for unit in variants}
        if len(core_tools) != 1:
            rejected.append({"card_id": card_id, "reason": "inconsistent_core_tool"})
            continue
        first = variants[0]
        by_name = {schema_name(schema): schema for schema in first["schemas"]}
        _, _, first_core, _ = first["location"]
        cards.append({
            "card_id": card_id, "core_tool": first["core_tool"],
            "core_schema": by_name.get(first["core_tool"]) or inferred_recorded_tool_schema(first["core_tool"], args_of(first_core)),
            "available_tools": schema_summary(first["schemas"]),
            "expected_aux_tools": sorted({unit["expected_tool"] for unit in variants}), "variants": variants,
        })
    if args.max_cards:
        cards = cards[:args.max_cards]

    mechanism_cache = out / "teacher_card_mechanisms"
    action_cache = out / "teacher_variant_actions"
    mechanism_cache.mkdir(exist_ok=True)
    action_cache.mkdir(exist_ok=True)
    mechanisms: dict[str, dict[str, Any]] = {}
    mechanism_failures: dict[str, str] = {}
    mechanism_feedback: dict[str, str] = {}

    def make_mechanism(card: dict[str, Any], feedback: str) -> tuple[str, dict[str, Any] | None, str | None]:
        cache = mechanism_cache / f"{card['card_id']}.json"
        try:
            raw = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else call_teacher(
                api_key, args.model, shared_mechanism_prompt(card, feedback), args.retries,
                disable_reasoning=args.disable_reasoning)
            value = validate_shared_mechanism(raw, card, {item["name"] for item in card["available_tools"]})
            cache.write_text(json.dumps({key: val for key, val in value.items() if not key.startswith("_")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return card["card_id"], value, None
        except Exception as exc:
            return card["card_id"], None, f"{type(exc).__name__}: {exc}"

    pending_cards = list(cards)
    for pass_index in range(args.retry_passes + 1):
        if not pending_cards:
            break
        if pass_index:
            time.sleep(args.retry_backoff * pass_index)
        current: dict[str, str] = {}
        processed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(make_mechanism, card, mechanism_feedback.get(card["card_id"], "")) for card in pending_cards]
            for future in concurrent.futures.as_completed(futures):
                card_id, value, error = future.result()
                processed += 1
                if error:
                    current[card_id] = error
                elif value is not None:
                    mechanisms[card_id] = value
                (out / "progress.json").write_text(json.dumps({"status": "running", "phase": "shared_mechanisms", "pass": pass_index, "completed_cards_this_pass": processed, "total_cards": len(cards), "successful_cards": len(mechanisms), "failed_cards_this_pass": len(current)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        mechanism_failures = current
        mechanism_feedback = current
        pending_cards = [card for card in cards if card["card_id"] in current]
        (out / "progress.json").write_text(json.dumps({"status": "running", "phase": "shared_mechanisms", "pass": pass_index, "total_cards": len(cards), "successful_cards": len(mechanisms), "failed_cards": len(current)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    units = [unit for card in cards if card["card_id"] in mechanisms for unit in card["variants"]]
    actions: dict[str, dict[str, Any]] = {}
    action_failures: dict[str, str] = {}
    action_feedback: dict[str, str] = {}

    def make_action(unit: dict[str, Any], feedback: str) -> tuple[str, dict[str, Any] | None, str | None]:
        cache = action_cache / f"{unit['probe_id']}.json"
        try:
            mechanism = mechanisms[unit["card_id"]]
            raw = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else call_teacher(
                api_key, args.model, per_variant_action_prompt(unit, mechanism, feedback), args.retries,
                disable_reasoning=args.disable_reasoning)
            value = validate_per_variant_action(raw, unit, mechanism)
            cache.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return unit["probe_id"], value, None
        except Exception as exc:
            return unit["probe_id"], None, f"{type(exc).__name__}: {exc}"

    pending_units = list(units)
    unit_by_id = {unit["probe_id"]: unit for unit in units}
    for pass_index in range(args.retry_passes + 1):
        if not pending_units:
            break
        if pass_index:
            time.sleep(args.retry_backoff * pass_index)
        current = {}
        processed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(make_action, unit, action_feedback.get(unit["probe_id"], "")) for unit in pending_units]
            for future in concurrent.futures.as_completed(futures):
                probe_id, value, error = future.result()
                processed += 1
                if error:
                    current[probe_id] = error
                elif value is not None:
                    actions[probe_id] = value
                (out / "progress.json").write_text(json.dumps({"status": "running", "phase": "per_probe_actions", "pass": pass_index, "completed_probes_this_pass": processed, "total_probes": len(units), "successful_probes": len(actions), "failed_probes_this_pass": len(current)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        action_failures = current
        action_feedback = current
        pending_units = [unit_by_id[probe_id] for probe_id in current]
        (out / "progress.json").write_text(json.dumps({"status": "running", "phase": "per_probe_actions", "pass": pass_index, "total_probes": len(units), "successful_probes": len(actions), "failed_probes": len(current)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fake_probes: list[dict[str, Any]] = []
    fake_labels: list[dict[str, Any]] = []
    real_probes: list[dict[str, Any]] = []
    real_labels: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for unit in units:
        action = actions.get(unit["probe_id"])
        if action is None:
            continue
        mechanism = mechanisms[unit["card_id"]]
        replacement = mechanism["replacement_tool"]
        replacement_name = str(replacement["name"])
        schemas = copy.deepcopy(unit["schemas"])
        if not replacement.get("reuse_existing"):
            schemas.append(copy.deepcopy(mechanism["_resolved_schema"]))
        real_probe = copy.deepcopy(unit["probe"])
        real_probe["trace_kind"] = "llm_parametric_real_paired"
        real_probe["tool_schemas"] = copy.deepcopy(schemas)
        fake_probe = copy.deepcopy(unit["probe"])
        fake_probe["trace_kind"] = "teacher_rewritten_core_fake"
        fake_probe["messages"] = rewrite_messages(unit["probe"], unit["location"], replacement_name, action["arguments"], action["thought"])
        fake_probe["tool_schemas"] = schemas
        real_label = copy.deepcopy(unit["label"])
        real_label.update({"original_probe_id": unit["probe_id"], "pair_role": "real", "paired_fake_core_tool": replacement_name, "original_core_tool": unit["core_tool"], "teacher_model": args.model, "thought_policy": "preserve_original_core_reason"})
        fake_label = copy.deepcopy(unit["label"])
        fake_label.update({"original_probe_id": unit["probe_id"], "pair_role": "fake", "fake_core_tool": replacement_name, "fake_core_arguments": action["arguments"], "fake_core_thought": action["thought"], "original_core_tool": unit["core_tool"], "teacher_model": args.model, "method_reason": mechanism["method_reason"], "input_binding": mechanism["input_binding"], "observation_contract": mechanism["observation_contract"], "thought_policy": "per_probe_generated_reason"})
        real_probes.append(real_probe); real_labels.append(real_label)
        fake_probes.append(fake_probe); fake_labels.append(fake_label)
        pairs.append({"probe_id": unit["probe_id"], "card_id": unit["card_id"], "benchmark": unit["benchmark"], "original_core_tool": unit["core_tool"], "fake_core_tool": replacement_name, "expected_tool": unit["expected_tool"], "expected_arguments": unit["expected_arguments"]})

    write_jsonl(out / "probes.jsonl", fake_probes)
    write_jsonl(out / "labels.jsonl", fake_labels)
    write_jsonl(out / "pairs.jsonl", pairs)
    failure_rows = rejected + [{"card_id": card_id, "reason": reason} for card_id, reason in sorted(mechanism_failures.items())] + [{"probe_id": probe_id, "card_id": unit_by_id[probe_id]["card_id"], "reason": reason} for probe_id, reason in sorted(action_failures.items())]
    write_jsonl(out / "rejected.jsonl", failure_rows)
    if args.paired_real_output_dir:
        real_out = Path(args.paired_real_output_dir)
        real_out.mkdir(parents=True, exist_ok=True)
        write_jsonl(real_out / "probes.jsonl", real_probes)
        write_jsonl(real_out / "labels.jsonl", real_labels)
        write_jsonl(real_out / "pairs.jsonl", pairs)
        (real_out / "summary.json").write_text(json.dumps({"mode": "jointly_built_real_side", "paired_probes": len(real_probes), "pairing_policy": "one shared mechanism per card; one action and Thought per probe"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"source_probes": len(labels), "requested_cards": len(cards), "shared_mechanism_cards": len(mechanisms), "requested_probes": len(units), "rewritten_probes": len(actions), "failed_mechanism_cards": len(mechanism_failures), "failed_actions": len(action_failures), "pre_teacher_rejections": dict(Counter(row["reason"] for row in rejected)), "teacher_model": args.model, "generation_mode": "shared_card_mechanism_per_probe_action", "observation_policy": "byte-for-byte reuse; no replacement action execution", "thought_policy": "each virtual probe receives its own teacher-written action rationale"}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "progress.json").write_text(json.dumps({"status": "complete", **summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(out / "retry_failures.jsonl", failure_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired real and virtual-Core probes from recorded release cards, without loading a benchmark.")
    parser.add_argument("--bench", choices=BENCHES)
    parser.add_argument("--teacher", choices=TEACHERS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--traces", "--trace-dir", dest="trace_source", type=Path,
                        help="Original trajectory JSON/JSONL file or directory.")
    parser.add_argument("--release-evidence", type=Path,
                        help="Optional JSON/JSONL aux_release evidence emitted while embedding.")
    parser.add_argument("--key-file", type=Path,
                        help="Optional local file containing the OpenRouter key; prefer OPENROUTER_API_KEY in automation.")
    parser.add_argument("--real-output-dir", type=Path)
    parser.add_argument("--fake-output-dir", type=Path)
    parser.add_argument("--work-real-dir", type=Path)
    parser.add_argument("--source-traces", "--source-trace-dir", dest="source_trace_dir", type=Path,
                        help="Optional original trajectories retained for audit provenance; probe construction uses each probe's recorded schemas.")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--variants-per-card", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-passes", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=15.0)
    parser.add_argument("--max-cards", type=int, default=0)
    parser.add_argument("--allow-text-observation", action="store_true",
                        help="Treat a recorded non-JSON tool observation as the exact object {\"text\": content}.")
    parser.add_argument(
        "--disable-reasoning",
        action="store_true",
        help="Request hosted models to disable reasoning for compact structured probe generation.",
    )
    parser.add_argument("--reuse-real-candidates", action="store_true", help="Reuse an existing real-candidate work directory unchanged.")
    args = parser.parse_args()
    if args.bench:
        if not args.teacher:
            parser.error("--bench requires --teacher")
        source = trace_dir(args.bench, args.teacher, "S", args.output_root)
        target = area_dir(args.bench, "probes", args.teacher, args.output_root) / "S"
        args.trace_source = args.trace_source or source / "standard_traces"
        args.release_evidence = args.release_evidence or source / "watermark_evidence"
        args.real_output_dir = args.real_output_dir or target / "real_probes"
        args.fake_output_dir = args.fake_output_dir or target / "fake_probes"
    for name in ("trace_source", "real_output_dir", "fake_output_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required without --bench")
    work = args.work_real_dir or (args.fake_output_dir.parent / "_real_parameter_candidates")
    if not (args.reuse_real_candidates and (work / "probes.jsonl").exists() and (work / "labels.jsonl").exists()):
        build_real_candidates(argparse.Namespace(
            trace_source=args.trace_source, release_evidence=args.release_evidence,
            output_dir=work, model=args.model, workers=args.workers,
            variants_per_card=args.variants_per_card, retries=args.retries, key_file=args.key_file,
            disable_reasoning=args.disable_reasoning, allow_text_observation=args.allow_text_observation,
        ))
    build_fake_pairs_grouped_legacy(argparse.Namespace(source_dir=work, output_dir=args.fake_output_dir, model=args.model, workers=args.workers, retries=args.retries, retry_passes=args.retry_passes, retry_backoff=args.retry_backoff, max_cards=args.max_cards, paired_real_output_dir=args.real_output_dir, source_trace_dir=args.source_trace_dir, key_file=args.key_file, disable_reasoning=args.disable_reasoning))


if __name__ == "__main__":
    main()
