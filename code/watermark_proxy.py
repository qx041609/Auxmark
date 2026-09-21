from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from openai import OpenAI
from model_config import auxiliary_api_key, auxiliary_base_url, teacher_api_key, teacher_base_url


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEACHER_MODEL = "deepseek/deepseek-chat"
DEFAULT_FILTER_MODEL = "qwen/qwen3-8b"
OUT_DIR = Path(os.getenv("AGENTWM_OUT_DIR", "/tmp/agentwm_proxy_unconfigured"))
FULL_TRACE_PATH = Path(os.getenv("AGENTWM_FULL_TRACE", OUT_DIR / "full_traces.jsonl"))
SKETCH_PATH = Path(os.getenv("AGENTWM_SKETCH_TRACE", OUT_DIR / "watermark_sketches.jsonl"))
AUX_START = "<agentwm_aux_candidates>"
AUX_END = "</agentwm_aux_candidates>"
JOINT_INSTRUCTION = f"""
Choose the normal core tool call exactly as you otherwise would. Whenever you make a core tool call,
also list up to __AGENTWM_K__ reasonable actions that could naturally follow it. These are brainstorming
candidates, so do not decide their later safety/relevance ranking; a downstream checker handles that.
The tool schemas supplied with this request are authoritative: use only
an exact supplied tool name and only argument names accepted by that tool's schema. Never invent a tool,
an argument, an entity value, or a replacement for the core call.

Before listing a candidate, check the current system instructions, user request, and workflow reflected in
the conversation. Propose an Aux only when inserting it at this exact point would not interrupt, reorder,
or contradict an explicitly required task procedure or mandatory state transition. If every otherwise
plausible follow-up would conflict with that procedure, return an empty candidates list.

In the assistant content, give one concise visible line beginning with `Thought:`, then always append:
{AUX_START}{{"candidates":[{{"candidate_id":"candidate_01","thought":"optional reason",
"tool":"available_tool","argument_plan":{{"argument_name":{{"source":"core_observation.field"}}}}}}]}}{AUX_END}
Use source references when an argument can come from the core result, core arguments, earlier observations,
or the user request. The private block is removed before the response is returned. Never alter the core tool
choice merely to make a candidate possible.
""".strip()

# Explicit structures injected into prompts so models (e.g. Llama) do not invent
# free-form text "tool calls" or free-form candidate lists.
CORE_TOOL_CALL_STRUCTURE = """
You are an agent that MUST use the API tool-calling interface when acting.

WRONG (never do this — plain text is NOT a tool call and will not execute):
  telecom_get_customer_by_phone(phone_number="123")
  Call tool X with arg=Y
  ```tool ... ```

RIGHT: use the request's tools / tool_calls mechanism only. Conceptually each
tool call has this data structure (names must match AVAILABLE tools exactly):
  {
    "name": "<exact_tool_name_from_tools_list>",
    "arguments": { "<arg_name>": <scalar_value>, ... }
  }

Rules:
1. If you need environment state, emit a real tool call (structured tool_calls),
   not a textual imitation of a call.
2. Tool name and argument keys must match the tools schema in this request.
3. Visible assistant text may contain a short Thought line only; do not put fake
   function-call syntax in content.
4. If you are done and need no tool, reply with normal assistant text only.
""".strip()

# Candidates are generated only after the core tool has returned a real
# observation.  Keeping this separate from the core completion prevents the
# model from guessing fields that do not exist in the eventual tool result.
CANDIDATE_INSTRUCTION = """
The primary action has completed and its real result is supplied below. Your job
is to propose optional post-core watermark candidates, not the normal next task
action. A downstream budget/scorer/checker will decide whether to inject one.

The request also supplies ORIGINAL_SYSTEM_INSTRUCTIONS verbatim. They are the
benchmark's authoritative task procedure; do not reinterpret, weaken, or
override them. Before proposing any candidate, check whether those instructions
explicitly prescribe an immediate successor, an exact action order, or an
uninterruptible state transition at this exact point. Only those explicit
immediate sequencing constraints block an Aux: if one applies, return
should_add_aux=false with an empty candidates list. Do not treat the mere fact
that other normal task steps remain to be done later as an immediate constraint;
an optional, non-interfering Aux may precede such later work. This is a
contextual judgment from the supplied instructions: never assume a fixed tool
name or benchmark-specific rule.

This distinction is mandatory: every returned candidate must be a reasonable,
read-only *supplementary follow-up to the Core action*. It may be useful and
may inspect or verify the real Core result, but it must not be necessary for
the current task itself: remove the candidate and the task objective, required
actions, and final task result must remain unchanged and fully achievable.
Ask: after the Core action has run, would this still be a sensible extra check
even though the agent can omit it without changing how the task is completed?
If yes, it is eligible. It is forbidden only when it is the explicitly required
immediate next step, a prerequisite that must run before any other action, or a
follow-up whose order is explicitly fixed by the task instructions. Do not
reject a candidate merely because it is related to a later normal task step.
Prefer a narrow verification, diagnostic, audit/status lookup, or contextual
inspection that naturally follows the Core action.

Every returned candidate must itself be runnable and correct: its tool must
exist, every required argument must be bindable from the supplied real context
(or be a concrete scalar), and it must not interfere with the primary task
result. Return up to {k} distinct candidates, but never pad the list with a
speculative, task-necessary, or invalid candidate. Return should_add_aux=false
whenever no candidate satisfies every condition above; an empty candidate list
is preferable to a normal task continuation.

Good candidates are small, plausible, non-destructive checks compatible with
the primary action result. They must be auxiliary behavior, never a replacement
for the primary action or the next task action. CRITICAL: stay in the same
tool domain/family as CORE_ACTION. If the core tool is a tau2 telecom tool, only
propose telecom tools; if airline, only airline; if retail, only retail; if
terminalbench, only terminalbench. Never cross from telecom to retail/airline or
any other unrelated domain just because those tools are also loaded. Prefer
candidates that are mutually different and complementary: inspect another
relevant object, verify a nearby fact, list available options, retrieve
status/context, or read a specific id mentioned by the primary result. Avoid
returning the same tool with effectively the same arguments as the primary
action unless the argument is meaningfully different.

For terminalbench, terminalbench_execute_bash is an allowed candidate. Prefer a
diagnostic or verification command that leaves the task's substantive result
unchanged (for example: pwd, ls/find, cat/sed/grep, git status/diff, or a test
command). Do not propose deletion, overwrite/redirection, package installation,
permission changes, process termination, or repository reset/cleanup commands.
For coding, correctness is semantic: the command in a candidate must actually
implement the one-line diagnostic/verification intent in `thought`; equivalent
commands are fine, but a command unrelated to that intent is not.
Execution-style verification is allowed, but must be bounded: write it as
`timeout 30s <command>` (or a shorter timeout), never background it, and never
start a persistent or interactive service. Plain inspection commands need no
timeout. A pure inline `python -c` analysis that only reads files, computes, and
prints a result is also allowed without a timeout; it is an inspection action,
not a task-running script.

=== OUR CANDIDATE DATA STRUCTURE (mandatory) ===
Return exactly ONE JSON object (no markdown fences, no extra text). Shape:

{{
  "should_add_aux": true,
  "reason": "short why these candidates are plausible",
  "candidates": [
    {{
      "candidate_id": "candidate_01",
      "thought": "one-line reason for this follow-up",
      "tool": "exact_tool_name_from_AVAILABLE_TOOLS",
      "argument_plan": {{
        "arg_name_1": {{"value": "concrete_scalar"}},
        "arg_name_2": {{"source": "core_observation.some_field"}}
      }}
    }},
    {{
      "candidate_id": "candidate_02",
      "thought": "another distinct follow-up",
      "tool": "another_exact_tool_name",
      "argument_plan": {{
        "id": {{"value": "L1001"}}
      }}
    }}
  ]
}}

Field rules:
- candidates contains one to {k} individually valid candidates, numbered consecutively from candidate_01.
- each candidate has EXACTLY: candidate_id, thought, tool, argument_plan.
- tool must be an exact name from AVAILABLE_TOOLS (same domain as CORE_ACTION).
- argument_plan keys must be that tool's real argument names.
- each argument_plan value is an object; preferred concrete form is {{"value": <scalar>}}.
  optional path form: {{"source":"core_observation.field"}} or
  {{"source":"core_action.arguments.x"}} or {{"source":"prior_observation.0.y"}} or
  {{"source":"user_request.text"}} or {{"source":"literal","value": <scalar>}}.
- do NOT invent a different schema (no "parameters", no free-text tool lines).

If and only if no valid candidate exists, return exactly:
{{"should_add_aux": false, "reason": "brief reason", "candidates": []}}
""".strip()


def now() -> int:
    return int(time.time())


def schedule_config_from_environment() -> Tuple[int, float, float, float]:
    """Read and validate the per-trace Aux scheduling configuration."""
    try:
        total_budget = int(os.getenv("AGENTWM_TOTAL_BUDGET", "10"))
        base_probability = float(os.getenv("AGENTWM_BASE_PROBABILITY", "0.1"))
        probability_increment = float(os.getenv("AGENTWM_PROBABILITY_INCREMENT", "0.1"))
        max_probability = float(os.getenv("AGENTWM_MAX_PROBABILITY", "1.0"))
    except ValueError as exc:
        raise RuntimeError(
            "AGENTWM_TOTAL_BUDGET must be an integer; scheduling probabilities must be numbers."
        ) from exc
    if total_budget < 0:
        raise RuntimeError("AGENTWM_TOTAL_BUDGET must be greater than or equal to 0.")
    probabilities = {
        "AGENTWM_BASE_PROBABILITY": base_probability,
        "AGENTWM_PROBABILITY_INCREMENT": probability_increment,
        "AGENTWM_MAX_PROBABILITY": max_probability,
    }
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()):
        raise RuntimeError("Scheduling probabilities must be finite values in the range [0, 1].")
    if base_probability > max_probability:
        raise RuntimeError("AGENTWM_BASE_PROBABILITY cannot exceed AGENTWM_MAX_PROBABILITY.")
    return total_budget, base_probability, probability_increment, max_probability


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def inject_system_message(messages: List[Dict[str, Any]], content: str) -> List[Dict[str, Any]]:
    """Append structure instructions to the first system message, or prepend one."""
    out = [dict(m) for m in messages]
    for message in out:
        if message.get("role") == "system":
            existing = str(message.get("content") or "")
            message["content"] = (existing + "\n\n" + content).strip() if existing else content
            return out
    return [{"role": "system", "content": content}, *out]


def write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


_TRACE_LOCK = threading.Lock()
_TRACE_SEQ = itertools.count(1)


def write_trace_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Write trace events with a process-local monotonic sequence number."""
    with _TRACE_LOCK:
        if "event_seq" not in row:
            row["event_seq"] = next(_TRACE_SEQ)
        write_jsonl(path, row)


def _first_balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def extract_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())

    balanced = _first_balanced_json_object(stripped)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else {}
        except Exception:
            continue
    return {}


def session_id_from_request(payload: Dict[str, Any], request: Request) -> str:
    for header in ("x-agentwm-session", "x-session-id", "x-agentmark-session"):
        if value := request.headers.get(header):
            return value.strip()
    seed = [
        str(message.get("content", ""))[:2000]
        for message in (payload.get("messages") or [])[:4]
        if message.get("role") in {"system", "user"}
    ]
    return hashlib.sha256(("\n".join(seed) or stable_json(payload.get("messages", [])[:2])).encode()).hexdigest()[:24]


def account_id_from_request(payload: Dict[str, Any], request: Request, session_id: str) -> str:
    for header in ("x-agentwm-account", "x-account-id", "x-user-id"):
        if value := request.headers.get(header):
            return value.strip()
    return str(payload.get("user") or os.getenv("AGENTWM_ACCOUNT_ID") or session_id)


def is_suspicious_request(payload: Dict[str, Any], request: Request) -> bool:
    header = request.headers.get("x-agentwm-suspicious")
    if header is not None:
        return header.lower() in {"1", "true", "yes"}
    return os.getenv("AGENTWM_SUSPICIOUS_BY_DEFAULT", "1").lower() in {"1", "true", "yes"}


def tool_function(tool: Dict[str, Any]) -> Dict[str, Any]:
    value = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return value if isinstance(value, dict) else {}


def tool_schemas(tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(fn["name"]): tool
        for tool in (tools or [])
        if (fn := tool_function(tool)).get("name")
    }


def tool_domain_family(name: Optional[str]) -> str:
    """Map a tool name to a coarse domain/family used for same-domain aux filtering.

    General-AgentBench exposes many MCP servers at once (tau2 airline/retail/telecom,
    terminalbench, search, ...). Aux candidates must stay in the same family as the
    just-executed core tool so a tau2 telecom turn cannot inject retail/airline actions.
    """
    text = str(name or "").strip()
    if not text:
        return ""
    if "__" in text:
        return text.split("__", 1)[0]
    known = (
        "telecom",
        "airline",
        "retail",
        "terminalbench",
        "mathhay",
        "search",
        "swebench",
        "browser",
        "filesystem",
    )
    for prefix in known:
        if text == prefix or text.startswith(prefix + "_"):
            return prefix
    return text.split("_", 1)[0]


# Only enforce hard same-domain filtering for multi-MCP families that are commonly
# co-loaded (especially tau2 airline/retail/telecom). Generic single-env tools like
# list_dir/read_file keep loose matching so unit tests and simple file agents work.
STRICT_DOMAIN_FAMILIES = {
    "telecom",
    "airline",
    "retail",
    "terminalbench",
    "mathhay",
    "search",
    "swebench",
    "browser",
    "filesystem",
}


def same_domain_family(core_tool: Optional[str], candidate_tool: Optional[str]) -> bool:
    core_family = tool_domain_family(core_tool)
    cand_family = tool_domain_family(candidate_tool)
    if not core_family or not cand_family:
        return True
    if core_family not in STRICT_DOMAIN_FAMILIES:
        return True
    return core_family == cand_family


def _tool_name_prefix(name: str) -> str:
    """Best-effort domain/family prefix for prioritization (telecom, retail, airline, ...)."""
    return tool_domain_family(name)


def compact_tool_catalog(
    tools: Optional[List[Dict[str, Any]]],
    core_tool: Optional[str] = None,
    max_chars: int = 12000,
    same_domain_only: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Build a compact, same-domain tool catalog for candidate prompts.

    Full OpenAI tool schemas for the multi-MCP agent often exceed the prompt budget.
    Truncating the full JSON then drops later domains (e.g. telecom), so models refuse
    or invent cross-domain follow-ups. Prefer compact name/arg catalogs restricted to
    the core tool's domain family (tau2 telecom core -> telecom tools only).
    """
    schemas = tool_schemas(tools)
    core_prefix = _tool_name_prefix(core_tool or "")
    domain_filtered = False
    if same_domain_only and core_prefix:
        same = {name: tool for name, tool in schemas.items() if same_domain_family(core_tool, name)}
        if same:
            schemas = same
            domain_filtered = True

    def priority(name: str) -> Tuple[int, str]:
        score = 0
        if core_tool and name == core_tool:
            score -= 1000
        name_prefix = _tool_name_prefix(name)
        if core_prefix and name_prefix == core_prefix:
            score -= 500
        if core_tool and name.startswith(str(core_tool).split("_")[0] + "_"):
            score -= 100
        # Mild preference for typical read-only verbs.
        for token in ("get_", "list_", "find_", "search_", "lookup_", "check_", "read_"):
            if token in name:
                score -= 10
                break
        return (score, name)

    ordered_names = sorted(schemas.keys(), key=priority)
    catalog: Dict[str, Dict[str, Any]] = {}
    omitted = 0
    for name in ordered_names:
        tool = schemas[name]
        fn = tool_function(tool)
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        props_in = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        props: Dict[str, Any] = {}
        for key, spec in props_in.items():
            if isinstance(spec, dict):
                props[str(key)] = {
                    k: spec.get(k)
                    for k in ("type", "enum", "description")
                    if spec.get(k) is not None
                }
                if isinstance(props[str(key)].get("description"), str):
                    props[str(key)]["description"] = props[str(key)]["description"][:80]
            else:
                props[str(key)] = {"type": type(spec).__name__}
        entry = {
            "required": list(params.get("required") or []),
            "properties": props,
        }
        desc = fn.get("description")
        if isinstance(desc, str) and desc.strip():
            entry["description"] = desc.strip()[:120]
        trial = dict(catalog)
        trial[name] = entry
        if len(stable_json(trial)) > max_chars and catalog:
            omitted += 1
            continue
        catalog[name] = entry
    meta = {
        "total_tools": len(tool_schemas(tools)),
        "domain_tools": len(schemas),
        "shown_tools": len(catalog),
        "omitted_tools": omitted,
        "same_domain_only": domain_filtered,
        "core_tool": core_tool,
        "core_prefix": core_prefix or None,
        "shown_names": list(catalog.keys()),
    }
    return catalog, meta


def message_text_content(message: Any) -> str:
    """Extract visible text from OpenAI-style chat messages, including reasoning models."""
    if message is None:
        return ""
    if isinstance(message, dict):
        data = message
    else:
        data = message.model_dump() if hasattr(message, "model_dump") else {}
        for attr in ("content", "reasoning", "reasoning_content", "refusal"):
            if attr not in data:
                try:
                    data[attr] = getattr(message, attr, None)
                except Exception:
                    pass
    for key in ("content", "reasoning_content", "reasoning", "refusal"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            joined = "\n".join(parts).strip()
            if joined:
                return joined
    return ""


def core_call(message: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    calls = message.get("tool_calls") or []
    if not calls:
        return None, {}, None
    call = calls[0]
    fn = tool_function(call)
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return str(fn.get("name")) if fn.get("name") else None, args if isinstance(args, dict) else {}, call.get("id")


def _reasoning_json_objects(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read complete JSON objects embedded after a reasoning provider's text."""
    text = message_text_content(message)
    decoder = json.JSONDecoder()
    objects: List[Dict[str, Any]] = []
    offset = 0
    while True:
        start = text.find("{", offset)
        if start < 0:
            return objects
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        offset = start + max(consumed, 1)


def normalise_reasoning_tool_call(response: Dict[str, Any], tools: Optional[List[Dict[str, Any]]]) -> None:
    """Convert one unambiguous reasoning-channel call to the OpenAI tool wire form.

    Certain provider/model combinations place a valid selected function JSON in
    ``message.reasoning`` while returning neither ``content`` nor
    ``tool_calls``.  The agent adapter can execute that action, but the
    watermark engine must see the same Core call to preserve paired semantics.
    This is a response-format compatibility shim only: it never chooses a
    tool, changes arguments, or relaxes the supplied tool schemas.
    """
    message = response.get("choices", [{}])[0].get("message")
    if not isinstance(message, dict) or message.get("tool_calls"):
        return
    schemas = tool_schemas(tools)
    if not schemas:
        return
    for value in reversed(_reasoning_json_objects(message)):
        name, arguments = value.get("name"), value.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            name, arguments = value.get("function_name"), value.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            matches: List[Tuple[str, Dict[str, Any]]] = []
            keys = set(value)
            for tool_name, schema in schemas.items():
                params = tool_function(schema).get("parameters", {})
                properties = params.get("properties", {}) if isinstance(params, dict) else {}
                required = set(params.get("required", []) if isinstance(params, dict) else [])
                if required.issubset(keys) and keys.issubset(set(properties)):
                    matches.append((tool_name, value))
            if len(matches) != 1:
                continue
            name, arguments = matches[0]
        valid, _ = final_call_check(name, arguments, tools)
        if not valid:
            continue
        message["tool_calls"] = [{
            "id": "call_reasoning_" + uuid.uuid4().hex[:12],
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }]
        message["content"] = None
        return


def parse_joint_content(content: str) -> Tuple[str, List[Dict[str, Any]]]:
    start, end = content.find(AUX_START), content.find(AUX_END)
    if start < 0:
        return content.strip(), []
    if end < start:
        return content[:start].strip(), []
    data = extract_json(content[start + len(AUX_START) : end].strip())
    candidates = data.get("candidates", [])
    visible = (content[:start] + content[end + len(AUX_END) :]).strip()
    return visible, [item for item in candidates if isinstance(item, dict)] if isinstance(candidates, list) else []


def ensure_thought(content: str, tool_name: Optional[str]) -> str:
    text = content.strip()
    if text.lower().startswith("thought:"):
        return text
    if text:
        return f"Thought: {text}"
    return f"Thought: I will call {tool_name} with the needed arguments to continue the task."


def load_safety_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Tool safety config must be a JSON object: {path}")
    return value


def tool_risk(name: str, config: Dict[str, Any]) -> str:
    exact = config.get("tools", {}).get(name, {}) if isinstance(config.get("tools"), dict) else {}
    if isinstance(exact, dict) and exact.get("risk"):
        return str(exact["risk"])
    # BFCL tool names contain a stable class/function slug followed by a hash.
    # Use the explicit class.function table when one is supplied, rather than
    # treating every bfcl_* tool as safe.  A missing entry intentionally falls
    # through to unknown, which keeps new BFCL functions fail-closed.
    bfcl_rules = config.get("bfcl_function_risks")
    if name.startswith("bfcl_") and isinstance(bfcl_rules, dict):
        def _slug(value: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip()).strip("_").lower()

        for class_name, functions in bfcl_rules.items():
            if not isinstance(functions, dict):
                continue
            class_slug = _slug(str(class_name))[:22]
            for function_name, risk in functions.items():
                prefix = f"bfcl_{class_slug}_{_slug(str(function_name))[:32]}_"
                if name.startswith(prefix):
                    return str(risk)
    for item in config.get("prefixes", []):
        if isinstance(item, dict) and name.startswith(str(item.get("prefix", ""))):
            return str(item.get("risk", "unknown"))
    return str(config.get("default_risk", "unknown"))


def basic_candidate_check(
    candidate: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]],
    safety: Dict[str, Any],
    core_tool: Optional[str] = None,
) -> Tuple[bool, str]:
    name = candidate.get("tool")
    if not name or str(name) not in tool_schemas(tools):
        return False, "tool_not_available"
    if core_tool and not same_domain_family(core_tool, str(name)):
        return False, "tool_cross_domain"
    # The terminalbench shell is admissible only after its concrete command
    # passes the non-interference check below.  A read-only allowlist made the
    # generator's most natural coding candidates unreachable.
    if str(name) != "terminalbench__terminalbench_execute_bash" and tool_risk(str(name), safety) != "read_only":
        return False, "tool_not_read_only"
    return True, "ok"


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def argument_source_shape_ok(spec: Any) -> bool:
    """Accept flexible argument_plan entries; binder only needs a resolvable scalar."""
    if _is_scalar(spec):
        return True
    if not isinstance(spec, dict):
        return False
    # Direct concrete value (optionally with ignored provenance hints).
    if "value" in spec and _is_scalar(spec.get("value")):
        return True
    source = spec.get("source")
    if not isinstance(source, str):
        # field alone as concrete value is allowed
        return "field" in spec and _is_scalar(spec.get("field"))
    if source == "literal":
        return "value" in spec and _is_scalar(spec.get("value"))
    root, _, _ = source.partition(".")
    if root not in {"core_observation", "core_action", "prior_observation", "user_request", "literal"}:
        # Treat bare non-path source strings as concrete values (legacy model output).
        return True
    allowed = {"source", "field", "value", "evidence_source"}
    if set(spec) - allowed:
        return False
    if "field" in spec and not _is_scalar(spec.get("field")):
        return False
    if "value" in spec and not _is_scalar(spec.get("value")):
        return False
    return True


def candidate_shape_ok(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    required = {"candidate_id", "thought", "tool", "argument_plan"}
    if set(candidate) != required:
        return False
    if not isinstance(candidate.get("candidate_id"), str) or not isinstance(candidate.get("thought"), str):
        return False
    if not isinstance(candidate.get("tool"), str):
        return False
    argument_plan = candidate.get("argument_plan")
    if not isinstance(argument_plan, dict):
        return False
    return all(isinstance(name, str) and argument_source_shape_ok(spec) for name, spec in argument_plan.items())


def candidate_response_shape_ok(value: Any) -> bool:
    """Validate the response envelope; candidates are validated individually.

    One malformed candidate must not discard its valid siblings.  This mirrors
    the collection adapters' per-call handling: keep executable calls and
    report only the malformed item that was skipped.
    """
    if not isinstance(value, dict):
        return False
    if set(value) != {"should_add_aux", "reason", "candidates"}:
        return False
    if not isinstance(value.get("should_add_aux"), bool) or not isinstance(value.get("reason"), str):
        return False
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        return False
    if value["should_add_aux"] is False:
        return candidates == []
    return True


def clamp_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def repeat_score(count: int) -> float:
    return 0.0 if count <= 0 else 0.6 if count < 5 else 1.0


def behavior_signature(core_tool: str, aux_tool: str) -> str:
    return f"{core_tool}|{aux_tool}|post_core"


def dotted_get(value: Any, path: str) -> Tuple[bool, Any]:
    current = value
    if not path:
        return True, current
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def parse_observation(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except Exception:
        return {"text": content}


def matching_observation(messages: List[Dict[str, Any]], call_id: Optional[str]) -> Any:
    for message in reversed(messages):
        if message.get("role") == "tool" and (not call_id or message.get("tool_call_id") == call_id):
            return parse_observation(message.get("content"))
    return None


def prior_observations(messages: List[Dict[str, Any]], exclude_call_id: Optional[str]) -> List[Any]:
    return [
        parse_observation(message.get("content"))
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id") != exclude_call_id
    ]


def user_text(messages: List[Dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")


def source_contains_value(sources: Dict[str, Any], value: str) -> bool:
    if not value:
        return False
    return value in stable_json(sources)


def refine_bound_value(value: Any, spec: Dict[str, Any]) -> Tuple[bool, Any]:
    field = spec.get("field")
    if field is None:
        return True, value
    field_text = str(field)
    if isinstance(value, str) and field_text in value:
        return True, field_text
    if isinstance(value, dict) and field_text in value:
        return True, value[field_text]
    if isinstance(value, list) and field_text.isdigit() and int(field_text) < len(value):
        return True, value[int(field_text)]
    if _is_scalar(field):
        return True, field
    return False, None


def extract_plan_scalar(argument: str, spec: Any, sources: Dict[str, Any]) -> Tuple[bool, Any, str]:
    """Take the concrete scalar from the candidate plan only.

    No context name-matching fallback. Path provenance is optional: if a path
    resolves, great; if not, keep any concrete value already in the plan
    (value/field/literal/source-as-value). Final gate is final_call_check
    (tool runnable), not provenance.
    """
    if _is_scalar(spec):
        return True, spec, "direct_value"
    if not isinstance(spec, dict):
        return False, None, f"invalid_source_plan:{argument}"

    # 1) Explicit value in plan — keep it, ignore whether evidence path is true.
    if _is_scalar(spec.get("value")):
        return True, spec["value"], "value"

    source = spec.get("source")
    # 2) Optional: try path; if it works, use resolved scalar.
    if isinstance(source, str) and source != "literal":
        root, _, path = source.partition(".")
        if root in sources:
            ok, value = dotted_get(sources[root], path)
            if ok:
                ok2, value = refine_bound_value(value, spec)
                if ok2 and _is_scalar(value):
                    tag = source if spec.get("field") is None else f"{source}#{spec.get('field')}"
                    return True, value, tag
        elif source not in {"core_observation", "core_action", "prior_observation", "user_request"}:
            # Model put the concrete arg in "source".
            return True, source, "source_as_value"

    # 3) field often is the concrete scalar the model wants (keep it).
    if _is_scalar(spec.get("field")):
        return True, spec["field"], "field"

    if source == "literal" and _is_scalar(spec.get("value")):
        return True, spec["value"], "literal"

    return False, None, f"no_concrete_arg:{argument}"


def bind_arguments(candidate: Dict[str, Any], pending: "PendingAux", messages: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, str], str]:
    """Build aux tool arguments from the candidate plan.

    Policy: only need a runnable tool call (name + schema-valid args).
    Do not invent values from context when paths fail; only keep scalars
    already present in the plan (or successfully read from a path).
    """
    observation = matching_observation(messages, pending.core_call_id)
    sources: Dict[str, Any] = {
        "core_observation": observation if observation is not None else {},
        "core_action": {"arguments": pending.core_arguments or {}},
        "prior_observation": prior_observations(messages, pending.core_call_id),
        "user_request": {"text": user_text(messages)},
    }
    bound: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}
    plan = candidate.get("argument_plan") or {}
    if not isinstance(plan, dict):
        return None, {}, "invalid_argument_plan"
    for argument, spec in plan.items():
        ok, value, tag = extract_plan_scalar(str(argument), spec, sources)
        if not ok:
            return None, provenance, tag
        bound[str(argument)] = value
        provenance[str(argument)] = tag
    return bound, provenance, "ok"


def schema_type_ok(value: Any, expected: Any) -> bool:
    types = expected if isinstance(expected, list) else [expected]
    checks = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
    return any(kind in checks and isinstance(value, checks[kind]) and not (kind in {"integer", "number"} and isinstance(value, bool)) for kind in types)


def final_call_check(tool: str, arguments: Dict[str, Any], tools: Optional[List[Dict[str, Any]]]) -> Tuple[bool, str]:
    schema = tool_schemas(tools).get(tool)
    if not schema:
        return False, "tool_not_available"
    params = tool_function(schema).get("parameters", {})
    required = params.get("required", []) if isinstance(params, dict) else []
    missing = [name for name in required if name not in arguments]
    if missing:
        return False, f"missing_required_args:{missing}"
    properties = params.get("properties", {}) if isinstance(params, dict) else {}
    for name, value in arguments.items():
        expected = properties.get(name, {}).get("type") if isinstance(properties.get(name), dict) else None
        if expected and not schema_type_ok(value, expected):
            return False, f"argument_type_mismatch:{name}"
    return True, "ok"


def noninterference_check(tool: str, arguments: Dict[str, Any]) -> Tuple[bool, str]:
    """Allow useful shell diagnostics while blocking task-state changes."""
    if tool != "terminalbench__terminalbench_execute_bash":
        return True, "ok"
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return False, "bash_command_missing"
    dangerous = (
        r"\brm\b", r"\bunlink\b", r"\bshred\b", r"\btruncate\b",
        r"\bmv\b", r"\bcp\b", r"\btouch\b", r"\bmkdir\b",
        r"\bchmod\b", r"\bchown\b", r"\bkill(all)?\b", r"\bpkill\b",
        r"\b(apt|apt-get|yum|dnf|pip|conda|npm)\s+(install|remove|uninstall|update)",
        r"\bgit\s+(reset|checkout|clean|restore)", r"\btee\b", r">>",
        r"(?<!\d)>(?![&])",
    )
    if any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in dangerous):
        return False, "bash_may_change_task_state"
    if re.search(r"(^|[;&|])\s*(nohup\s+|.*\s&\s*$|.*\b(screen|tmux)\b)", command, flags=re.IGNORECASE):
        return False, "bash_may_background_or_persist"
    # Pure inline Python data analysis (read -> compute -> print) is a useful
    # inspection behavior.  It should not be penalized merely for using an
    # interpreter.  Writes, process spawning, network/service use, and input
    # make it an execution-style action instead.
    inline_python = bool(re.search(r"\bpython(?:3)?\s+-c\s+", command, flags=re.IGNORECASE))
    inline_side_effects = (
        r"\bopen\s*\([^)]*,\s*['\"][wax+]", r"\bpathlib\..*\.write_",
        r"\bos\.(remove|unlink|rename|replace|mkdir|makedirs|chmod|chown|system)\b",
        r"\b(subprocess|Popen|run\s*\(|call\s*\()", r"\b(requests|socket|http\.server)\b",
        r"\b(input|serve|listen|daemon)\s*\(",
    )
    inline_has_side_effects = any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in inline_side_effects)
    if inline_python and inline_has_side_effects:
        return False, "bash_inline_analysis_may_change_task_state"
    readonly_inline_analysis = inline_python and not inline_has_side_effects
    # Running tests or existing programs can be useful watermark behavior, but
    # an unbounded invocation may hang a benchmark worker indefinitely.  Keep
    # those only when the teacher makes the runtime budget explicit.
    execution = re.search(
        r"(^|[;&|])\s*(?:cd\s+\S+\s*&&\s*)?(?:python(?:3)?|node|ruby|perl|bash|sh|make|pytest|go\s+test|cargo\s+test)\b",
        command,
        flags=re.IGNORECASE,
    )
    bounded = re.search(r"(^|[;&|])\s*timeout\s+(?:-k\s+\S+\s+)?(?:[1-9]|[1-5][0-9])s?\s+", command)
    if execution and not bounded and not readonly_inline_analysis:
        return False, "bash_execution_requires_timeout"
    return True, "ok"


@dataclass
class PendingAux:
    candidate: Dict[str, Any]
    core_message: Dict[str, Any]
    core_tool: str
    core_arguments: Dict[str, Any]
    core_call_id: Optional[str]
    probability_before: float
    sample: float
    scores: Dict[str, Any]


@dataclass
class PendingCore:
    """A core tool call awaiting its real environment observation."""
    core_message: Dict[str, Any]
    core_tool: str
    core_arguments: Dict[str, Any]
    core_call_id: Optional[str]


@dataclass
class ReleasedAux:
    # ``call_id`` is the private evidence ID.  The recorder may replace it in
    # the agent-visible response and store that value in ``public_call_id``.
    # Both release and execution evidence always retain the private ID.
    call_id: str
    core_round_index: int
    tool: str
    arguments: Dict[str, Any]
    public_call_id: Optional[str] = None


@dataclass
class SessionState:
    account_id: str
    trace_id: str
    suspicious: bool
    total_budget: int
    remaining_budget: int
    current_probability: float
    core_round_index: int = 0
    successful_watermark_count: int = 0
    pending_core: Optional[PendingCore] = None
    released_aux: Optional[ReleasedAux] = None


class WatermarkEngine:
    def __init__(self) -> None:
        teacher_key = teacher_api_key()
        if not teacher_key:
            raise RuntimeError("Configure a teacher API key before starting the proxy.")
        filter_key = auxiliary_api_key()
        if not filter_key:
            raise RuntimeError("Configure AGENTWM_AUX_API_KEY for the filter/safety client before starting the proxy.")
        # Teacher and the auxiliary reviewer can have distinct latency profiles.  In particular,
        # a reasoning teacher may need longer for the internal candidate JSON
        # response, while Qwen review calls should retain their normal bound.
        timeout = float(os.getenv("AGENTWM_AUX_TIMEOUT", os.getenv("AGENTWM_OPENROUTER_TIMEOUT", "60")))
        teacher_timeout = float(os.getenv("AGENTWM_TEACHER_TIMEOUT", str(timeout)))
        # Keep watermark selection/injection untouched, but make transport
        # behaviour configurable for bounded evaluation runs.  The OpenAI
        # client's implicit retries can otherwise turn a single stalled
        # upstream request into several minutes with no result-row flush.
        max_retries = int(os.getenv("AGENTWM_AUX_MAX_RETRIES", os.getenv("AGENTWM_OPENROUTER_MAX_RETRIES", "2")))
        teacher_max_retries = int(os.getenv("AGENTWM_TEACHER_MAX_RETRIES", str(max_retries)))
        self.client = OpenAI(
            api_key=teacher_key,
            base_url=teacher_base_url(OPENROUTER_BASE_URL),
            timeout=teacher_timeout,
            max_retries=teacher_max_retries,
        )
        # A collection can explicitly opt into a provider failover without
        # changing any existing run.  The fallback reuses the isolated
        # OpenRouter credential already configured for the auxiliary model.
        self.fallback_client: Optional[OpenAI] = None
        self.fallback_teacher_model = os.getenv("AGENTWM_TEACHER_FALLBACK_MODEL", "").strip()
        if os.getenv("AGENTWM_TEACHER_FALLBACK_ENABLED", "0").lower() in {"1", "true", "yes"}:
            fallback_key = os.getenv("AGENTWM_TEACHER_FALLBACK_API_KEY", "").strip() or filter_key
            fallback_url = os.getenv("AGENTWM_TEACHER_FALLBACK_URL", "").strip() or auxiliary_base_url(OPENROUTER_BASE_URL)
            if not self.fallback_teacher_model:
                raise RuntimeError("Set AGENTWM_TEACHER_FALLBACK_MODEL when enabling teacher failover.")
            self.fallback_client = OpenAI(
                api_key=fallback_key,
                base_url=fallback_url,
                timeout=teacher_timeout,
                max_retries=0,
            )
        self.filter_client = OpenAI(
            api_key=filter_key,
            base_url=auxiliary_base_url(OPENROUTER_BASE_URL),
            timeout=timeout,
            max_retries=max_retries,
        )
        self.teacher_model = os.getenv("AGENTWM_TEACHER_MODEL", DEFAULT_TEACHER_MODEL)
        self.filter_model = os.getenv("AGENTWM_FILTER_MODEL", DEFAULT_FILTER_MODEL)
        self.model_alias = os.getenv("AGENTWM_MODEL_ALIAS", "agentwm-watermarked-teacher")
        self.k = int(os.getenv("AGENTWM_K", "3"))
        if self.k < 1:
            raise RuntimeError("AGENTWM_K must be greater than or equal to 1.")
        (
            self.total_budget,
            self.base_probability,
            self.probability_increment,
            self.max_probability,
        ) = schedule_config_from_environment()
        self.secret_seed = os.getenv("AGENTWM_SECRET_SEED", "change-me-before-collection")
        # The embedding endpoint attaches a conservative policy derived from
        # the *current request's* tool schemas.  Keep this fail-closed fallback
        # only for direct/legacy callers that do not use the endpoint.
        self.safety = {"tools": {}, "prefixes": [], "default_risk": "unknown"}
        self.sessions: Dict[str, SessionState] = {}
        self.behavior_history: Dict[str, Dict[str, int]] = {}
        self._load_behavior_history()

    def _load_behavior_history(self) -> None:
        if not SKETCH_PATH.exists():
            return
        with SKETCH_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                account = event.get("account_id")
                signature = event.get("behavior_signature")
                if event.get("event") == "aux_release" and account and isinstance(signature, str):
                    bucket = self.behavior_history.setdefault(str(account), {})
                    bucket[signature] = bucket.get(signature, 0) + 1

    def state(self, session_id: str, account_id: str, suspicious: bool) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(account_id, session_id, suspicious, self.total_budget, self.total_budget, self.base_probability)
        return self.sessions[session_id]

    def random_sample(self, state: SessionState) -> float:
        raw = f"{self.secret_seed}|{state.account_id}|{state.trace_id}|{state.core_round_index}|v1"
        number = int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big")
        return number / float(2**64)

    def update_failure(self, state: SessionState) -> None:
        state.current_probability = min(self.max_probability, state.current_probability + self.probability_increment)

    @staticmethod
    def _is_teacher_quota_or_capacity_error(exc: Exception) -> bool:
        """Return true only for a primary-provider quota/capacity response."""
        status = getattr(exc, "status_code", None)
        text = str(exc).lower()
        if status in {402, 429}:
            return True
        return status == 403 and any(marker in text for marker in (
            "quota", "balance", "credit", "insufficient", "billing", "limit", "额度", "余额",
        ))

    def teacher_completion(self, kwargs: Dict[str, Any]) -> Any:
        """Call ModelArk first, using OpenRouter only after quota/capacity failure."""
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self.fallback_client is None or not self._is_teacher_quota_or_capacity_error(exc):
                raise
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["model"] = self.fallback_teacher_model
            # ModelArk's thinking extension is not accepted by OpenRouter.
            # Preserve requested effort in OpenRouter's provider-neutral wire
            # format, and omit it for explicitly disabled candidate calls.
            extra_body = fallback_kwargs.pop("extra_body", None)
            if isinstance(extra_body, dict):
                effort = extra_body.get("reasoning_effort")
                thinking = extra_body.get("thinking")
                if effort in {"low", "medium", "high"} and not (
                    isinstance(thinking, dict) and thinking.get("type") == "disabled"
                ):
                    fallback_kwargs["extra_body"] = {"reasoning": {"effort": effort, "exclude": False}}
            write_trace_jsonl(FULL_TRACE_PATH, {
                "event": "teacher_provider_failover",
                "from_provider": "ModelArk",
                "to_provider": "OpenRouter",
                "fallback_model": self.fallback_teacher_model,
                "status_code": getattr(exc, "status_code", None),
                "error": str(exc),
                "created": now(),
            })
            return self.fallback_client.chat.completions.create(**fallback_kwargs)

    def call_teacher(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate only the normal core response; never ask it for Aux here."""
        default_effort = os.getenv("AGENTWM_DEFAULT_REASONING_EFFORT", "").strip().lower()
        has_reasoning_config = any(
            payload.get(name) is not None
            for name in ("reasoning", "thinking", "reasoning_effort")
        )
        if default_effort == "none" and not has_reasoning_config:
            # Kimi K2.5 on ModelArk may default to thinking when the field is
            # omitted.  ``none`` must therefore be an explicit opt-out rather
            # than merely the absence of an opt-in.
            payload = dict(payload)
            payload["thinking"] = {"type": "disabled"}
        elif default_effort in {"low", "medium", "high"} and not has_reasoning_config:
            # Some benchmark clients (notably Tau2 through LiteLLM) cannot
            # express ModelArk's provider-specific thinking fields.  Let the
            # launcher opt in explicitly without changing legacy runs.
            payload = dict(payload)
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = default_effort
        messages = list(payload.get("messages") or [])
        tools = payload.get("tools")
        # Proactively inject our tool-call data structure so models do not emit
        # plain-text fakes like `tool_name(arg=...)` instead of API tool_calls.
        if tools:
            names = list(tool_schemas(tools).keys())
            preview = names[:40]
            structure = (
                CORE_TOOL_CALL_STRUCTURE
                + "\n\nAVAILABLE tool names in this request (use exact strings):\n"
                + stable_json(preview)
                + (f"\n... and {len(names) - len(preview)} more" if len(names) > len(preview) else "")
            )
            messages = inject_system_message(messages, structure)
        kwargs: Dict[str, Any] = {
            "model": self.teacher_model,
            "messages": messages,
            "temperature": payload.get("temperature", 0.0),
        }
        for name in ("tools", "tool_choice", "max_tokens"):
            if payload.get(name) is not None:
                kwargs[name] = payload[name]
        # These provider extensions aren't represented by the installed SDK.
        # ``extra_body`` preserves them on the wire without changing watermark
        # selection: OpenRouter uses ``reasoning`` while ModelArk uses
        # ``thinking`` plus ``reasoning_effort``.
        extra_body = {
            name: payload[name]
            for name in ("reasoning", "thinking", "reasoning_effort")
            if payload.get(name) is not None
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = self.teacher_completion(kwargs).model_dump()
        normalise_reasoning_tool_call(response, tools)
        return response

    def generate_candidates(self, state: SessionState, payload: Dict[str, Any], pending: PendingCore) -> Tuple[List[Dict[str, Any]], str]:
        """Generate candidates after the primary tool result is present."""
        observation = matching_observation(payload.get("messages") or [], pending.core_call_id)
        if observation is None:
            return [], "core_observation_missing"
        catalog, catalog_meta = compact_tool_catalog(payload.get("tools"), core_tool=pending.core_tool, max_chars=12000)
        all_messages = payload.get("messages") or []
        # The benchmark procedure often lives in the initial system message and
        # may have fallen out of the short conversation window by later turns.
        # Forward it verbatim to the candidate model; this is context transport,
        # not a benchmark-specific rule or a rewritten prompt.
        original_system = [
            {"role": "system", "content": str(message.get("content") or "")}
            for message in all_messages
            if isinstance(message, dict) and message.get("role") == "system"
            and str(message.get("content") or "").strip()
        ]
        recent_context = all_messages[-8:]
        user_prompt = (
            "ORIGINAL_SYSTEM_INSTRUCTIONS (authoritative; preserve their exact workflow):\n"
            f"{stable_json(original_system)[:16000]}\n\n"
            "The fields below are the real post-core context. You may and should "
            "use these exact source names in argument_plan: core_action, "
            "core_observation, prior_observation, user_request, literal.\n\n"
            "CORE_ACTION is the primary tool call that just ran:\n"
            f"{stable_json({'tool': pending.core_tool, 'arguments': pending.core_arguments})[:4000]}\n\n"
            "CORE_OBSERVATION is the real result returned by that primary tool. "
            "It is present and valid; do not claim it is missing:\n"
            f"{stable_json(observation)[:10000]}\n\n"
            "RECENT_CONTEXT, for prior_observation/user_request references:\n"
            f"{stable_json(recent_context)[:8000]}\n\n"
            "AVAILABLE_TOOLS (our catalog data structure: tool_name -> "
            "{{required: [...], properties: {{name: type, ...}}}}; same-domain only):\n"
            f"{stable_json(catalog)}\n\n"
            f"(catalog coverage: shown={catalog_meta.get('shown_tools')} "
            f"domain={catalog_meta.get('domain_tools')} total={catalog_meta.get('total_tools')}; "
            f"same_domain_only={catalog_meta.get('same_domain_only')}; "
            f"core_prefix={catalog_meta.get('core_prefix')})\n\n"
            "Respond with OUR CANDIDATE DATA STRUCTURE only (see system message). "
            "Example skeleton with concrete value form:\n"
            f"{stable_json({'should_add_aux': True, 'reason': 'example', 'candidates': [{'candidate_id': 'candidate_01', 'thought': 'verify related id', 'tool': (list(catalog.keys())[0] if catalog else 'exact_tool_name'), 'argument_plan': {'id': {'value': 'example_id'}}}]})}\n\n"
            "Generate candidates now from this context. Do not say the core action "
            "or core observation is missing; they are provided explicitly above."
        )
        prompt = [
            {"role": "system", "content": CANDIDATE_INSTRUCTION.format(k=self.k)},
            {"role": "user", "content": user_prompt},
        ]
        # Reasoning models (e.g. Kimi K2.5) may spend many tokens before JSON.
        cand_max_tokens = int(os.getenv("AGENTWM_CANDIDATE_MAX_TOKENS", "2500"))
        try:
            candidate_kwargs: Dict[str, Any] = {
                "model": self.teacher_model,
                "messages": prompt,
                "temperature": 0.0,
                "max_tokens": cand_max_tokens,
            }
            # Candidate generation is a small, strict JSON serialization job,
            # not an agent reasoning turn. Kimi K2.5 can otherwise consume a
            # long hidden-thinking budget and hit the transport timeout before
            # emitting JSON. This opt-in affects only the internal candidate
            # request; normal Core responses keep native teacher thinking.
            if os.getenv("AGENTWM_CANDIDATE_DISABLE_THINKING", "0").lower() in {"1", "true", "yes"}:
                candidate_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            response = self.teacher_completion(candidate_kwargs)
            choice = response.choices[0] if response.choices else None
            message = choice.message if choice is not None else None
            raw_content = message_text_content(message)
            finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None
            value = extract_json(raw_content)
            reason = str(value.get("reason") or ("empty_model_response" if not raw_content else "teacher_declined"))
            candidates = value.get("candidates", [])
            raw_candidate_count = len(candidates) if isinstance(candidates, list) else 0
            schema_ok = candidate_response_shape_ok(value)
            should_add_aux = value.get("should_add_aux")
            # Hard gate: only real MCP tool names from the same-domain catalog.
            allowed_tools = set(catalog.keys())
            kept: List[Dict[str, Any]] = []
            dropped: List[Dict[str, Any]] = []
            candidate_list = list(candidates) if schema_ok and should_add_aux is True else []
            for index, cand in enumerate(candidate_list, start=1):
                if not candidate_shape_ok(cand):
                    dropped.append({
                        "candidate_id": cand.get("candidate_id") if isinstance(cand, dict) else None,
                        "tool": cand.get("tool") if isinstance(cand, dict) else None,
                        "reason": "candidate_schema_invalid",
                    })
                    continue
                if cand.get("candidate_id") != f"candidate_{index:02d}":
                    dropped.append({
                        "candidate_id": cand.get("candidate_id"),
                        "tool": cand.get("tool"),
                        "reason": "candidate_id_out_of_sequence",
                    })
                    continue
                tool_name = str(cand.get("tool") or "")
                if tool_name in allowed_tools and same_domain_family(pending.core_tool, tool_name):
                    kept.append(cand)
                else:
                    dropped.append({
                        "candidate_id": cand.get("candidate_id"),
                        "tool": tool_name,
                        "reason": (
                            "tool_not_in_domain_catalog"
                            if tool_name not in allowed_tools
                            else "tool_cross_domain"
                        ),
                    })
            candidate_list = kept
            write_trace_jsonl(FULL_TRACE_PATH, {
                "event": "candidate_generation",
                "session_id": state.trace_id,
                "core_round_index": state.core_round_index,
                "parsed": bool(value),
                "schema_ok": schema_ok,
                "should_add_aux": should_add_aux,
                "raw_candidate_count": raw_candidate_count,
                "candidate_count": len(candidate_list),
                "candidates": candidate_list,
                "raw_candidates": candidates if isinstance(candidates, list) else None,
                "dropped_candidates": dropped or None,
                "raw_response": raw_content[:4000] if not schema_ok else None,
                "finish_reason": finish_reason,
                "tool_catalog": catalog_meta,
                "core_domain": tool_domain_family(pending.core_tool),
                "reason": reason,
                "created": now(),
            })
            if not schema_ok:
                return [], "candidate_schema_invalid" if raw_content else "empty_model_response"
            if should_add_aux is False:
                return [], reason
            if not candidate_list:
                return [], "no_valid_candidates" if dropped else reason
            return candidate_list, reason
        except Exception as exc:
            write_trace_jsonl(FULL_TRACE_PATH, {
                "event": "candidate_generation_error",
                "session_id": state.trace_id,
                "core_round_index": state.core_round_index,
                "error": str(exc),
                "created": now(),
            })
            return [], "candidate_generation_error"

    def runtime_safety(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Read the endpoint-provided per-tool-table policy without exposing it to the teacher."""
        policy = payload.get("_agentwm_safety")
        return policy if isinstance(policy, dict) else self.safety

    def score_candidates(self, state: SessionState, payload: Dict[str, Any], core_message: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        core_tool, core_arguments, core_call_id = core_call(core_message)
        safety = self.runtime_safety(payload)
        valid = []
        filter_rows = []
        for candidate in candidates[: self.k]:
            ok, reason = basic_candidate_check(candidate, payload.get("tools"), safety, core_tool=core_tool)
            if ok and core_tool:
                draft = PendingAux(
                    candidate, core_message, core_tool, core_arguments, core_call_id,
                    0.0, 0.0, {},
                )
                bound, _, bind_reason = bind_arguments(candidate, draft, payload.get("messages") or [])
                if bound is None:
                    ok, reason = False, bind_reason
                else:
                    ok, reason = final_call_check(str(candidate.get("tool")), bound, payload.get("tools"))
                    if ok:
                        ok, reason = noninterference_check(str(candidate.get("tool")), bound)
            filter_rows.append({
                "candidate_id": candidate.get("candidate_id"),
                "tool": candidate.get("tool"),
                "accepted": ok,
                "reason": reason,
            })
            if ok:
                valid.append(candidate)
        if filter_rows:
            write_trace_jsonl(FULL_TRACE_PATH, {"event": "candidate_filter", "session_id": state.trace_id, "core_round_index": state.core_round_index, "rows": filter_rows, "created": now()})
        if not valid or not core_tool:
            return []
        prompt = [
            {"role": "system", "content": "Rank already-validated optional post-core tool candidates. Return JSON only. Do not decide whether to inject; always score every candidate. Relevance means natural and task-related but non-essential. Reliability means the candidate is semantically correct for its stated Thought and concrete arguments; for terminalbench, judge whether the bash command genuinely performs that diagnostic/verification intent without changing task state. Equivalent shell commands are acceptable. Scores are numbers in [0,1]."},
            {"role": "user", "content": f"Context:\n{stable_json((payload.get('messages') or [])[-8:])[:8000]}\nCore:\n{stable_json(core_message)[:3000]}\nCandidates:\n{stable_json(valid)}\nReturn: {{\"scores\":[{{\"candidate_id\":\"...\",\"relevance\":0.0,\"reliability\":0.0,\"reason\":\"...\"}}]}}"},
        ]
        try:
            response = self.filter_client.chat.completions.create(model=self.filter_model, messages=prompt, temperature=0.0, max_tokens=600)
            rows = extract_json(response.choices[0].message.content or "").get("scores", [])
        except Exception as exc:
            write_trace_jsonl(FULL_TRACE_PATH, {"event": "candidate_scoring_error", "session_id": state.trace_id, "error": str(exc), "created": now()})
            rows = []
        by_id = {str(row.get("candidate_id")): row for row in rows if isinstance(row, dict)}
        scored: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        history = self.behavior_history.setdefault(state.account_id, {})
        for index, candidate in enumerate(valid):
            candidate_id = str(candidate.get("candidate_id") or f"candidate_{index + 1:02d}")
            model_score = by_id.get(candidate_id, {})
            relevance = clamp_score(model_score.get("relevance"))
            reliability = clamp_score(model_score.get("reliability"))
            signature = behavior_signature(core_tool, str(candidate.get("tool")))
            repeated = repeat_score(history.get(signature, 0))
            total = 0.35 * relevance + 0.25 * reliability + 0.40 * repeated
            score = {"candidate_id": candidate_id, "relevance": relevance, "reliability": reliability, "repeat": repeated, "total": total, "behavior_signature": signature, "reason": model_score.get("reason")}
            scored.append((candidate, score))
            write_trace_jsonl(FULL_TRACE_PATH, {"event": "candidate_score", "session_id": state.trace_id, "candidate": candidate, "score": score, "accepted": True, "created": now()})
        return sorted(scored, key=lambda item: item[1]["total"], reverse=True)

    def make_aux_response(self, candidate: Dict[str, Any], arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        call_id = f"call_aux_{uuid.uuid4().hex[:10]}"
        thought = ensure_thought(str(candidate.get("thought") or ""), str(candidate.get("tool")))
        response = {"id": f"chatcmpl-agentwm-{uuid.uuid4().hex}", "object": "chat.completion", "created": now(), "model": self.model_alias, "choices": [{"index": 0, "message": {"role": "assistant", "content": thought, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": candidate["tool"], "arguments": json.dumps(arguments, ensure_ascii=False)}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        return response, call_id

    def handle_post_core(self, state: SessionState, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pending_core = state.pending_core
        if not pending_core:
            return None
        if matching_observation(payload.get("messages") or [], pending_core.core_call_id) is None:
            return None

        state.pending_core = None
        probability_before = state.current_probability
        sample = self.random_sample(state)
        budget_available = state.remaining_budget > 0
        attempt_aux = state.suspicious and budget_available and sample < probability_before
        if not attempt_aux:
            self.update_failure(state)
            reason = "budget_exhausted" if state.suspicious and not budget_available else "schedule_miss"
            write_trace_jsonl(FULL_TRACE_PATH, {"event": "aux_skipped", "session_id": state.trace_id, "core_round_index": state.core_round_index, "sample": sample, "probability_before": probability_before, "probability_after": state.current_probability, "remaining_budget": state.remaining_budget, "reason": reason, "created": now()})
            return None

        candidates, candidate_reason = self.generate_candidates(state, payload, pending_core)
        scored = self.score_candidates(state, payload, pending_core.core_message, candidates)
        if not scored:
            self.update_failure(state)
            event = "aux_declined" if candidate_reason != "candidate_generation_error" and not candidates else "aux_cancel"
            reason = candidate_reason if event == "aux_declined" else "no_acceptable_candidate"
            write_trace_jsonl(FULL_TRACE_PATH, {"event": event, "session_id": state.trace_id, "core_round_index": state.core_round_index, "reason": reason, "probability_after": state.current_probability, "created": now()})
            return None

        candidate, scores = scored[0]
        pending = PendingAux(candidate, pending_core.core_message, pending_core.core_tool, pending_core.core_arguments, pending_core.core_call_id, probability_before, sample, scores)
        arguments, provenance, reason = bind_arguments(pending.candidate, pending, payload.get("messages") or [])
        if arguments is None:
            valid, final_reason = False, reason
        else:
            valid, final_reason = final_call_check(str(pending.candidate.get("tool")), arguments, payload.get("tools"))
            if valid:
                valid, final_reason = noninterference_check(str(pending.candidate.get("tool")), arguments)
        state.pending_aux = None
        if not valid or arguments is None:
            self.update_failure(state)
            write_trace_jsonl(FULL_TRACE_PATH, {"event": "aux_cancel", "session_id": state.trace_id, "core_round_index": state.core_round_index, "reason": final_reason, "probability_after": state.current_probability, "created": now()})
            return None
        response, call_id = self.make_aux_response(pending.candidate, arguments)
        before = state.remaining_budget
        state.remaining_budget -= 1
        state.successful_watermark_count += 1
        state.current_probability = self.base_probability
        signature = str(pending.scores["behavior_signature"])
        history = self.behavior_history.setdefault(state.account_id, {})
        history[signature] = history.get(signature, 0) + 1
        sketch = {"event": "aux_release", "session_id": state.trace_id, "account_id": state.account_id, "core_round_index": state.core_round_index, "call_id": call_id, "aux_tool": pending.candidate.get("tool"), "aux_arguments": arguments, "argument_provenance": provenance, "aux_semantic_role": pending.candidate.get("thought"), "behavior_signature": signature, "scores": pending.scores, "sample": pending.sample, "probability_before": pending.probability_before, "probability_after": state.current_probability, "remaining_budget_before": before, "remaining_budget_after": state.remaining_budget, "sequence": "post_core", "created": now()}
        sketch["core"] = {
            "tool": pending.core_tool,
            "arguments": pending.core_arguments,
            "observation": matching_observation(payload.get("messages") or [], pending.core_call_id),
        }
        sketch["core_domain"] = tool_domain_family(pending.core_tool)
        sketch["aux_domain"] = tool_domain_family(str(pending.candidate.get("tool") or ""))
        sketch["same_domain"] = same_domain_family(pending.core_tool, str(pending.candidate.get("tool") or ""))
        write_trace_jsonl(FULL_TRACE_PATH, sketch)
        write_trace_jsonl(SKETCH_PATH, sketch)
        state.released_aux = ReleasedAux(call_id, state.core_round_index, str(pending.candidate["tool"]), arguments)
        return response

    def record_released_aux_observation(self, state: SessionState, payload: Dict[str, Any]) -> None:
        """Append the real environment result for an Aux that was already released."""
        released = state.released_aux
        if not released:
            return
        # The executor only sees the neutral public ID, while private evidence
        # keeps ``call_aux_*``.  Match the public ID but write the private ID so
        # raw release/execution events are directly pairable.
        observed_call_id = released.public_call_id or released.call_id
        observation = matching_observation(payload.get("messages") or [], observed_call_id)
        if observation is None:
            return
        event = {
            "event": "aux_execution",
            "session_id": state.trace_id,
            "account_id": state.account_id,
            "core_round_index": released.core_round_index,
            "call_id": released.call_id,
            "aux_tool": released.tool,
            "aux_arguments": released.arguments,
            "aux_observation": observation,
            "execution_success": True,
            "created": now(),
        }
        write_trace_jsonl(FULL_TRACE_PATH, event)
        write_trace_jsonl(SKETCH_PATH, event)
        state.released_aux = None

    def handle(self, payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        session_id = session_id_from_request(payload, request)
        account_id = account_id_from_request(payload, request, session_id)
        state = self.state(session_id, account_id, is_suspicious_request(payload, request))
        self.record_released_aux_observation(state, payload)
        if response := self.handle_post_core(state, payload):
            return response

        state.core_round_index += 1
        response = self.call_teacher(payload)
        message = response.get("choices", [{}])[0].get("message", {})
        core_tool, core_arguments, core_call_id = core_call(message)
        if core_tool:
            message["content"] = ensure_thought(str(message.get("content") or ""), core_tool)
            state.pending_core = PendingCore(message, core_tool, core_arguments, core_call_id)
        elif state.suspicious and state.remaining_budget > 0:
            self.update_failure(state)
        write_trace_jsonl(FULL_TRACE_PATH, {"event": "core_response", "session_id": session_id, "account_id": account_id, "core_round_index": state.core_round_index, "message": message, "schedule": {"suspicious": state.suspicious, "deferred_until_core_observation": bool(core_tool), "probability": state.current_probability, "remaining_budget": state.remaining_budget}, "created": now()})
        response["model"] = self.model_alias
        return response


app = FastAPI(title="AgentWM Watermark Proxy v2")
_ENGINE: Optional[WatermarkEngine] = None


def engine() -> WatermarkEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = WatermarkEngine()
    return _ENGINE


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    return {"object": "list", "data": [{"id": engine().model_alias, "object": "model", "owned_by": "agentwm"}]}


@app.post("/v1/chat/completions")
def chat_completions(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
    try:
        return engine().handle(payload, request)
    except Exception as exc:
        write_trace_jsonl(FULL_TRACE_PATH, {"event": "proxy_error", "error": str(exc), "created": now()})
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8010")))
