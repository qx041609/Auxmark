#!/usr/bin/env python3
"""One user-facing entry for AgentWM safety preparation and proxy serving.

The watermark algorithm itself remains in ``watermark_proxy.py``.  This file
only prepares its runtime configuration, exposes the OpenAI-compatible
endpoint, and records the agent-observed trace/evidence after every turn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, HTTPException, Request

import watermark_proxy as core
from model_config import auxiliary_api_key, auxiliary_base_url, teacher_api_key, teacher_base_url


HERE = Path(__file__).resolve().parent
RECORD_OUT_DIR = Path(os.getenv("AGENTWM_OUT_DIR", "/tmp/agentwm_embedding_unconfigured"))
app = FastAPI(title="AgentWM Embedding Recorder")
_SAFETY_CACHE: Dict[str, Dict[str, Any]] = {}
_SAFETY_CACHE_LOCK = threading.Lock()


def normalise_tools(value: Any) -> list[dict[str, Any]]:
    """Accept the OpenAI ``tools`` list directly from a new agent request."""
    if isinstance(value, dict):
        value = value.get("tools", [])
    if not isinstance(value, list):
        raise RuntimeError("Tool file must be a JSON list or an object containing a tools list.")
    tools = [item for item in value if isinstance(item, dict) and isinstance((item.get("function") or item).get("name"), str)]
    if not tools:
        raise RuntimeError("The request contains no OpenAI function schemas.")
    return tools


def tool_schema_fingerprint(tools: list[dict[str, Any]]) -> str:
    raw = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def supplied_safety_policy(tools: list[dict[str, Any]], fingerprint: str) -> Dict[str, Any] | None:
    """Use a pre-processed matching policy if the caller supplied one.

    ``AGENTWM_SAFETY_PATH`` accepts either one JSON policy or a directory of
    ``<tool-schema-fingerprint>.json`` records previously emitted by this
    endpoint.  A policy is accepted only when every current tool has an
    explicit classification; an old or partial table never silently applies to
    a new tool schema.
    """
    configured = os.getenv("AGENTWM_SAFETY_PATH", "").strip()
    if not configured:
        return None
    path = Path(configured)
    candidates = [path / f"{fingerprint}.json"] if path.is_dir() else [path]
    names = {str((tool.get("function") or tool).get("name")) for tool in tools}
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            record = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        if record.get("tool_schema_fingerprint") and record.get("tool_schema_fingerprint") != fingerprint:
            continue
        policy = record.get("policy") if isinstance(record.get("policy"), dict) else record
        classified = policy.get("tools") if isinstance(policy.get("tools"), dict) else {}
        if names and all(isinstance(classified.get(name), dict) and classified[name].get("risk") in {"read_only", "unknown"} for name in names):
            return policy
    return None


def resolve_key(key_file: Path | None, local: bool = False) -> str:
    if local:
        return "local"
    value = teacher_api_key()
    if value:
        return value
    if key_file and key_file.exists():
        value = key_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    raise RuntimeError("Configure AGENTWM_TEACHER_API_KEY or the private teacher provider file, or pass --key-file.")


def probability_argument(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number in [0, 1]") from exc
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be a finite value in [0, 1]")
    return parsed


def nonnegative_integer_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer greater than or equal to 0") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def positive_integer_argument(value: str) -> int:
    parsed = nonnegative_integer_argument(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than or equal to 1")
    return parsed


def chat_json(endpoint: str, api_key: str, model: str, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:800]}") from exc
    content = str(data.get("choices", [{}])[0].get("message", {}).get("content") or "")
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("Safety-policy model did not return JSON.")
    value = json.loads(content[start:end + 1])
    if not isinstance(value, dict):
        raise RuntimeError("Safety-policy model returned a non-object.")
    return value


def classify_tool_safety(
    tools: list[dict[str, Any]], endpoint: str, api_key: str, model: str, max_tokens: int,
) -> dict[str, Any]:
    """Keep the existing conservative safety prompt, but consume live schemas."""
    compact = []
    for tool in tools:
        function = tool.get("function") or tool
        compact.append({"name": function.get("name"), "description": function.get("description"), "parameters": function.get("parameters")})
    prompt = f"""Classify every supplied agent tool conservatively for a non-interfering auxiliary-action watermark.

Return strict JSON only, with exactly this shape:
{{"tools": {{"exact_tool_name": {{"risk": "read_only|unknown"}}}}, "prefixes": [], "default_risk": "unknown"}}

Mark read_only ONLY when the schema and description clearly indicate a query,
lookup, list, read, status, or bounded diagnostic that cannot change external
state. Mark unknown for every write, delete, create, execute, shell, payment,
message, login, ordering, deployment, or ambiguous tool. Every supplied name
must occur exactly once under tools. Do not invent names or prefixes.

TOOLS={json.dumps(compact, ensure_ascii=False)}"""
    response = chat_json(endpoint, api_key, model, [
        {"role": "system", "content": "You are a conservative tool-safety classifier. Return JSON only."},
        {"role": "user", "content": prompt},
    ], max_tokens)
    returned = response.get("tools") if isinstance(response.get("tools"), dict) else {}
    names = {str((tool.get("function") or tool).get("name")) for tool in tools}
    policy = {"tools": {name: {"risk": "unknown"} for name in names}, "prefixes": [], "default_risk": "unknown"}
    for name, row in returned.items():
        if name in names and isinstance(row, dict) and row.get("risk") == "read_only":
            policy["tools"][name] = {"risk": "read_only"}
    return policy


def serve_proxy(args: argparse.Namespace) -> None:
    if args.base_probability > args.max_probability:
        raise RuntimeError("--base-probability cannot exceed --max-probability.")
    secret_seed = (args.secret_seed or os.getenv("AGENTWM_SECRET_SEED", "")).strip()
    if not secret_seed:
        raise RuntimeError("Set --secret-seed or AGENTWM_SECRET_SEED before starting the embedding endpoint.")
    env = os.environ.copy()
    env.update({
        # Teacher traffic uses neutral provider-independent names.  Do not
        # overwrite OPENROUTER_* here: Qwen filter/safety remains on
        # OpenRouter and its credential is intentionally separate.
        "AGENTWM_TEACHER_API_KEY": resolve_key(args.key_file, local=args.local_teacher),
        "AGENTWM_TEACHER_BASE_URL": args.teacher_url.rstrip("/"),
        "AGENTWM_TEACHER_MODEL": args.teacher_model,
        "AGENTWM_FILTER_MODEL": args.filter_model or args.teacher_model,
        "AGENTWM_SAFETY_MODEL": args.safety_model or args.filter_model or args.teacher_model,
        "AGENTWM_SAFETY_MAX_TOKENS": str(args.safety_max_tokens),
        "AGENTWM_SAFETY_PATH": str(args.safety_path) if args.safety_path else "",
        "AGENTWM_MODEL_ALIAS": args.model_alias,
        "AGENTWM_OUT_DIR": str(args.out_dir),
        "AGENTWM_SECRET_SEED": secret_seed,
        "AGENTWM_BASE_PROBABILITY": str(args.base_probability),
        "AGENTWM_PROBABILITY_INCREMENT": str(args.probability_increment),
        "AGENTWM_MAX_PROBABILITY": str(args.max_probability),
        "AGENTWM_TOTAL_BUDGET": str(args.total_budget),
        "AGENTWM_K": str(args.candidates_per_core),
    })
    runtime_python = args.python or os.getenv("AGENTWM_PYTHON") or sys.executable
    if not Path(runtime_python).exists():
        runtime_python = sys.executable
    # The recorder forwards every request unchanged to watermark_proxy while
    # continuously mirroring the agent-observed trace and evidence to disk.
    command = [runtime_python, "-m", "uvicorn", "embedding:app", "--app-dir", str(HERE),
               "--host", args.host, "--port", str(args.port)]
    subprocess.run(command, env=env, check=True)


# ---- Runtime recorder.  It wraps, but never edits, watermark_proxy.py. ----

_AUX_CALL_ID_MAP: Dict[str, Dict[str, str]] = {}


def runtime_safety_policy(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Parse one new live tool table once, then reuse its conservative policy.

    This replaces a hand-maintained, benchmark-specific safety JSON.  The
    policy is private runtime metadata: it is attached only to the call into
    ``watermark_proxy`` and is never passed to the teacher or agent.
    """
    raw_tools = payload.get("tools") or []
    if not raw_tools:
        return {"tools": {}, "prefixes": [], "default_risk": "unknown"}, "no_tools"
    tools = normalise_tools(raw_tools)
    fingerprint = tool_schema_fingerprint(tools)
    with _SAFETY_CACHE_LOCK:
        cached = _SAFETY_CACHE.get(fingerprint)
    if cached is not None:
        return cached, fingerprint

    policy = supplied_safety_policy(tools, fingerprint)
    source = "provided"
    if policy is None:
        source = "llm_generated"
        safety_key = auxiliary_api_key()
        if not safety_key:
            raise RuntimeError(
                "AGENTWM_AUX_API_KEY is required for the tool-safety classifier."
            )
        policy = classify_tool_safety(
            tools,
            auxiliary_base_url(),
            safety_key,
            os.getenv("AGENTWM_SAFETY_MODEL") or os.getenv("AGENTWM_FILTER_MODEL") or core.engine().teacher_model,
            int(os.getenv("AGENTWM_SAFETY_MAX_TOKENS", "4000")),
        )
    with _SAFETY_CACHE_LOCK:
        _SAFETY_CACHE[fingerprint] = policy
    _atomic_json(RECORD_OUT_DIR / "safety_policies" / f"{fingerprint}.json", {
        "schema": "agentwm_runtime_tool_safety_v1",
        "tool_schema_fingerprint": fingerprint,
        "tools": tools,
        "policy": policy,
        "source": source,
        "generated_at": int(time.time()),
    })
    return policy, fingerprint


def _neutral_call_id(raw_id: str, name: str, arguments: Any, index: int, salt: str = "") -> str:
    """Use an ordinary OpenAI call id in the agent-visible response."""
    seed = json.dumps([salt, raw_id, name, arguments, index], sort_keys=True, ensure_ascii=False)
    return "call_" + hashlib.sha256(seed.encode()).hexdigest()[:16]


def _normalise_tool_call(call: Dict[str, Any], index: int) -> Dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    name = str(function.get("name") or call.get("name") or call.get("tool") or "unknown_tool")
    arguments = function.get("arguments", call.get("arguments", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    raw_id = str(call.get("id") or call.get("tool_call_id") or "")
    call_id = raw_id if raw_id and not raw_id.startswith("call_aux_") else _neutral_call_id(raw_id, name, arguments, index)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True)}}


def standard_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep agent history unchanged except for watermark-only call identifiers."""
    output: List[Dict[str, Any]] = []
    remap: Dict[str, str] = {}
    outstanding: List[str] = []
    for index, raw in enumerate(messages):
        role = str(raw.get("role") or "").lower()
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        if role == "assistant":
            calls = []
            for call_index, call in enumerate(raw.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                previous_id = str(call.get("id") or call.get("tool_call_id") or "")
                normalised = _normalise_tool_call(call, index * 100 + call_index)
                if previous_id:
                    remap[previous_id] = normalised["id"]
                calls.append(normalised)
            message: Dict[str, Any] = {"role": "assistant", "content": str(raw.get("content") or "")}
            if calls:
                message["tool_calls"] = calls
                outstanding.extend(call["id"] for call in calls)
            output.append(message)
            continue
        if role == "tool":
            call_id = str(raw.get("tool_call_id") or raw.get("call_id") or raw.get("id") or "")
            call_id = remap.get(call_id, call_id) or (outstanding.pop(0) if outstanding else f"call_{index:04d}")
            content = raw.get("content", raw.get("observation", raw.get("result", "")))
            output.append({"role": "tool", "tool_call_id": call_id, "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)})
            continue
        output.append({"role": role, "content": str(raw.get("content") or "")})
    return output


def standard_quality(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    calls = [call["id"] for message in messages if message["role"] == "assistant" for call in message.get("tool_calls", [])]
    observations = {message.get("tool_call_id") for message in messages if message["role"] == "tool"}
    return {"assistant_turns": sum(message["role"] == "assistant" for message in messages), "tool_calls": len(calls),
            "paired_tool_calls": sum(call_id in observations for call_id in calls),
            "unpaired_tool_calls": sum(call_id not in observations for call_id in calls),
            "trainable": bool(calls or any(message["role"] == "assistant" for message in messages)) and all(call_id in observations for call_id in calls)}


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _session_file_id(session_id: str) -> str:
    return hashlib.sha256(session_id.encode()).hexdigest()[:16]


def _normalise_returned_call_ids(session_id: str, response: Dict[str, Any]) -> None:
    """Give every agent-visible call a unique opaque ID and preserve linkage.

    Some compatible providers emit a function name such as
    ``functions.get_details_by_id`` as the call ID on every round.  Tau2 then
    receives duplicate IDs in one conversation, which makes a later tool
    observation ambiguous.  BFCL/SWE pair calls and observations after
    normalisation; do the same at the embedding boundary and also update the
    proxy's pending Core ID to the exact public value.

    Private ``call_aux_*`` IDs remain only in the watermark evidence stream.
    """
    message = response.get("choices", [{}])[0].get("message") or {}
    mappings = _AUX_CALL_ID_MAP.setdefault(session_id, {})
    state = core.engine().sessions.get(session_id)
    for index, call in enumerate(message.get("tool_calls") or []):
        raw_id = str(call.get("id") or "")
        function = call.get("function") or {}
        round_index = state.core_round_index if state is not None else 0
        neutral = _neutral_call_id(
            raw_id,
            str(function.get("name") or "tool"),
            function.get("arguments"),
            index,
            salt=f"{session_id}|{round_index}",
        )
        call["id"] = neutral
        if raw_id.startswith("call_aux_"):
            mappings[raw_id] = neutral
        if state and state.released_aux and state.released_aux.call_id == raw_id:
            # Keep the private ID for both evidence events.  The executor sees
            # only this neutral ID, which is used solely to match its returned
            # observation.
            state.released_aux.public_call_id = neutral
        if state and state.pending_core and state.pending_core.core_call_id == raw_id:
            # ``handle_post_core`` must look for the same ID that Tau2 places
            # on the real ToolMessage; otherwise the observation exists but
            # the watermark candidate stage cannot see it.
            state.pending_core.core_call_id = neutral


def _evidence_for(session_id: str) -> List[Dict[str, Any]]:
    mappings = _AUX_CALL_ID_MAP.get(session_id, {})
    reverse = {public: private for private, public in mappings.items()}
    rows: List[Dict[str, Any]] = []
    if not core.SKETCH_PATH.exists():
        return rows
    for line in core.SKETCH_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("session_id") != session_id or row.get("event") not in {"aux_release", "aux_execution"}:
            continue
        row = dict(row)
        private = str(row.get("call_id") or "")
        public = mappings.get(private, private)
        if private in mappings:
            row["original_call_id"] = private
            row["returned_call_id"] = public
            row["call_id"] = public
        elif private in reverse:
            row["original_call_id"] = reverse[private]
            row["returned_call_id"] = private
        rows.append(row)
    return rows


def _record_session(payload: Dict[str, Any], request: Request, response: Dict[str, Any]) -> None:
    session_id = core.session_id_from_request(payload, request)
    message = dict(response.get("choices", [{}])[0].get("message") or {})
    message["role"] = "assistant"
    messages = [*list(payload.get("messages") or []), message]
    tools = list(payload.get("tools") or [])
    benchmark = str(payload.get("benchmark") or "unknown")
    domain = str(payload.get("domain") or "unknown")
    task_id = str(payload.get("task_id") or session_id)
    stamp = int(time.time())
    file_id = _session_file_id(session_id)
    base = {"session_id": session_id, "benchmark": benchmark, "domain": domain, "task_id": task_id,
            "teacher_model": core.engine().teacher_model, "tool_schemas": tools, "updated": stamp}
    _atomic_json(RECORD_OUT_DIR / "watermarked_traces" / f"{file_id}.json", {"schema": "agentwm_watermarked_trace_v1", **base, "messages": messages})
    normalised = standard_messages(messages)
    raw = json.dumps([benchmark, domain, task_id, normalised], sort_keys=True, ensure_ascii=False)
    _atomic_json(RECORD_OUT_DIR / "standard_traces" / f"{file_id}.json", {
        "schema": "agentwm_distillation_trace_v2", **base,
        "trace_id": hashlib.sha256(raw.encode()).hexdigest()[:24], "trace_kind": "watermarked",
        "messages": normalised, "quality": standard_quality(normalised),
    })
    _atomic_json(RECORD_OUT_DIR / "watermark_evidence" / f"{file_id}.json", {"session_id": session_id, "updated": stamp, "events": _evidence_for(session_id)})


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    return core.models()


@app.post("/v1/aux-observation")
def aux_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Record an already-returned Aux observation without generating a Core."""
    session_id = str(payload.get("session_id") or "")
    call_id = str(payload.get("tool_call_id") or "")
    if not session_id or not call_id:
        raise HTTPException(status_code=400, detail="session_id and tool_call_id are required")
    state = core.engine().sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown session")
    if state.released_aux is None:
        # A terminal Core is also executed by Tau2 after the simulator stops.
        # It has no private evidence to write, but is not an error.
        return {"status": "not_aux"}
    expected_call_id = state.released_aux.public_call_id or state.released_aux.call_id
    if call_id != expected_call_id:
        raise HTTPException(
            status_code=409,
            detail={"reason": "aux_tool_call_id_mismatch", "expected": expected_call_id, "received": call_id},
        )
    core.engine().record_released_aux_observation(
        state,
        {"messages": [{"role": "tool", "tool_call_id": call_id, "content": payload.get("observation", "")}]} ,
    )
    if state.released_aux is not None:
        raise HTTPException(status_code=409, detail="tool_call_id does not match the released Aux")
    evidence = RECORD_OUT_DIR / "watermark_evidence" / f"{_session_file_id(session_id)}.json"
    try:
        row = json.loads(evidence.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        row = {"session_id": session_id}
    row["events"] = _evidence_for(session_id)
    row["updated"] = int(time.time())
    _atomic_json(evidence, row)
    return {"status": "recorded"}


@app.get("/v1/aux-status")
def aux_status(session_id: str) -> Dict[str, Any]:
    """Return a read-only pairing audit for one local collection session."""
    state = core.engine().sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown session")
    events = _evidence_for(session_id)
    released = {str(row.get("call_id") or "") for row in events if row.get("event") == "aux_release"}
    executed = {str(row.get("call_id") or "") for row in events if row.get("event") == "aux_execution"}
    return {
        "pending_call_id": (
            (state.released_aux.public_call_id or state.released_aux.call_id)
            if state.released_aux else None
        ),
        "unpaired_release_ids": sorted(call_id for call_id in released if call_id and call_id not in executed),
        "released": len(released),
        "executed": len(executed),
    }


@app.post("/v1/chat/completions")
def chat_completions(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
    try:
        session_id = core.session_id_from_request(payload, request)
        safety, _ = runtime_safety_policy(payload)
        core_payload = dict(payload)
        core_payload["_agentwm_safety"] = safety
        response = core.chat_completions(core_payload, request)
        _normalise_returned_call_ids(session_id, response)
        _record_session(payload, request, response)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        core.write_trace_jsonl(core.FULL_TRACE_PATH, {"event": "recording_error", "error": str(exc), "created": int(time.time())})
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the AgentWM embedding endpoint.")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Start the OpenAI-compatible AgentWM embedding proxy.")
    serve.add_argument("--teacher-url", default=teacher_base_url(),
                       help="OpenAI-compatible API base URL; use a local vLLM/SGLang URL for a local teacher.")
    serve.add_argument("--teacher-model", required=True)
    serve.add_argument("--filter-model")
    serve.add_argument("--key-file", type=Path)
    serve.add_argument("--local-teacher", action="store_true")
    serve.add_argument("--safety-model", help="Optional model for runtime tool-safety parsing; defaults to --filter-model then --teacher-model.")
    serve.add_argument("--safety-max-tokens", type=int, default=4000)
    serve.add_argument("--safety-path", type=Path, help="Optional processed safety JSON, or a directory containing <tool-schema-fingerprint>.json. It is used only when it fully matches the request tools; otherwise the endpoint calls --safety-model.")
    serve.add_argument("--bench", choices=("BFCL", "SWEbench", "Tau2"))
    serve.add_argument("--teacher", choices=("GPT", "Kimi"))
    serve.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    serve.add_argument("--out-dir", type=Path)
    serve.add_argument("--secret-seed", help="Private watermark seed; alternatively set AGENTWM_SECRET_SEED in the environment.")
    serve.add_argument("--model-alias", default="agentwm-watermarked-teacher")
    serve.add_argument("--base-probability", type=probability_argument, default=0.1,
                       help="Initial probability and the probability restored after an Aux release (default: 0.1).")
    serve.add_argument("--probability-increment", type=probability_argument, default=0.1,
                       help="Probability added after a turn without an Aux release (default: 0.1).")
    serve.add_argument("--max-probability", type=probability_argument, default=1.0,
                       help="Upper bound for the scheduling probability (default: 1.0).")
    serve.add_argument("--total-budget", type=nonnegative_integer_argument, default=10,
                       help="Maximum Aux releases allowed in each trace (default: 10).")
    serve.add_argument("--candidates-per-core", type=positive_integer_argument, default=3)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8010)
    serve.add_argument("--python", help="Python environment containing FastAPI/OpenAI; defaults to AGENTWM_PYTHON or the existing agentwm environment.")
    serve.set_defaults(handler=serve_proxy)

    args = parser.parse_args()
    if args.command == "serve":
        if args.bench:
            if not args.teacher:
                parser.error("--bench requires --teacher")
            args.out_dir = args.out_dir or args.output_root / args.bench / "trace" / args.teacher / "_collection_work" / "embedding"
        if args.out_dir is None:
            parser.error("--out-dir is required without --bench")
    args.handler(args)


if __name__ == "__main__":
    main()
