#!/usr/bin/env python3
"""SWE-bench adapter: candidates, collection, D_c, and robustness data.

This is a benchmark adapter only.  Watermark scheduling, candidate filtering,
and evidence recording remain in ``code/embedding.py`` and ``code/watermark_proxy.py``.
Each accepted row has a unique task id and one fresh Docker Compose project.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import yaml


BASH = "terminalbench__terminalbench_execute_bash"
READ = "terminalbench__terminalbench_read_file"
WRITE = "terminalbench__terminalbench_write_file"
LIST = "terminalbench__terminalbench_list_dir"
FINISH = "terminalbench__terminalbench_finish"

# Exact public TerminalBench MCP table used by the earlier collection.  This
# adapter never selects an Aux tool itself: it transmits this table unchanged
# to watermark_proxy.py, which retains the only candidate/safety decision.
TOOLS = [
    {"type": "function", "function": {"name": BASH, "description": "Execute a bash command in the Terminal-Bench Docker container. The command runs in the task working directory. Only one bash command may execute at a time.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute."}, "timeout": {"type": ["number", "null"], "description": "Optional timeout in seconds (default: 120)."}, "wait_time_sec": {"type": "number", "description": "Time to wait after command execution before returning.", "default": 0.0}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": READ, "description": "Read the contents of a file in the Docker container.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Absolute path to the file to read."}, "start_line": {"type": ["integer", "null"]}, "end_line": {"type": ["integer", "null"]}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": WRITE, "description": "Write content to a file in the Docker container.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "append": {"type": "boolean", "default": False}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": LIST, "description": "List contents of a directory in the Docker container.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_depth": {"type": "integer", "default": 2}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": FINISH, "description": "Signals completion of the current task after success or a technical limitation.", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}},
]
TOOL_NAMES = {BASH, READ, WRITE, LIST, FINISH}
TOOL_ALIASES = {
    "execute_bash": BASH, "terminalbench_execute_bash": BASH, "terminalbench__execute_bash": BASH,
    "read_file": READ, "terminalbench_read_file": READ, "terminalbench__read_file": READ,
    "write_file": WRITE, "terminalbench_write_file": WRITE, "terminalbench__write_file": WRITE,
    "list_dir": LIST, "terminalbench_list_dir": LIST, "terminalbench__list_dir": LIST,
    "finish": FINISH, "terminalbench_finish": FINISH, "terminalbench__finish": FINISH,
}


def legacy_search_alias_to_bash(raw_name: str, arguments: Any) -> dict[str, Any] | None:
    """Translate common terminal-search aliases into the declared bash tool.

    Some tool-use checkpoints emit ``grep``/``find_file`` despite the exact
    table exposing only execute_bash.  These are not new capabilities: each
    form has one unambiguous, quoted shell equivalent using the already
    declared TerminalBench bash tool.  Unknown aliases remain hard errors.
    """
    if not isinstance(arguments, dict):
        return None
    short = raw_name.rsplit("__", 1)[-1].removeprefix("terminalbench_")
    root = arguments.get("path") or arguments.get("directory") or arguments.get("root") or "."
    if not isinstance(root, str):
        return None
    if short in {"grep", "search_file"}:
        pattern = arguments.get("pattern") or arguments.get("query") or arguments.get("search_term")
        if not isinstance(pattern, str):
            return None
        return {"command": f"grep -RIn --exclude-dir=.git -- {shlex.quote(pattern)} {shlex.quote(root)}"}
    if short in {"find_file", "search_files"}:
        needle = arguments.get("name") or arguments.get("filename") or arguments.get("query") or arguments.get("pattern")
        if not isinstance(needle, str):
            return None
        return {"command": f"find {shlex.quote(root)} -type f -name {shlex.quote(needle)}"}
    return None


def recover_bash_command(arguments: str) -> dict[str, Any] | None:
    """Recover a uniquely-delimited command from harmless JSON wrapper noise."""
    match = re.search(r'"command"\s*:\s*"((?:\\.|[^"\\])*)"', arguments, flags=re.DOTALL)
    if not match:
        return None
    try:
        command = json.loads('"' + match.group(1) + '"')
    except json.JSONDecodeError:
        return None
    return {"command": command} if isinstance(command, str) else None

# SWE-Bench supplies GitHub issue text, which needs only execution framing to
# become a repository-agent task.  This contains no benchmark answer, tool
# substitution, or watermark instruction.
SWE_AGENT_SYSTEM_PROMPT = """You are an autonomous software-engineering agent.
The user message is a real issue for the repository mounted in your container.
Resolve it in that repository; do not answer it as a discussion or propose a
hypothetical patch.

Use an engineering workflow: inspect the repository and relevant code, reason
about the failure, make the source change, and run focused checks when feasible.
Do not stop merely because you have inspected files or formed a plan. Continue
using the tools until you have implemented the best concrete fix available.
Call finish only after making the change and checking it, or after a genuine
technical limitation prevents further work; give a short factual summary then.
Use only the declared tools."""

def system_prompt_with_tools() -> str:
    """Give the model the exact executable table in both supported channels.

    Providers receive ``TOOLS`` as OpenAI function declarations, while the
    explicit copy below makes the permitted names unambiguous to models that
    reason in text before emitting a call.  The table is generated from the
    same object used for execution, so the two representations cannot drift.
    """
    return "\n\n".join((
        SWE_AGENT_SYSTEM_PROMPT,
        "The following is the complete executable TOOL_LIST. Do not invent, rename, or assume any other tools. "
        "Use an exact `name` from this list for every tool call. In particular, `find_file`, `search_file`, and `grep` "
        "are not tools. To locate files or search text, call the declared execute_bash tool with a shell `find` or `grep` command.",
        "<tool_list>\n" + json.dumps(TOOLS, ensure_ascii=False, separators=(",", ":")) + "\n</tool_list>",
    ))


def add_reasoning_config(payload: dict[str, Any], effort: str, protocol: str) -> None:
    """Attach the provider-native deep-thinking controls to one request."""
    if effort == "none":
        return
    if protocol == "modelark":
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = effort
    elif protocol == "openrouter":
        payload["reasoning"] = {"effort": effort, "exclude": False}
    else:
        raise ValueError(f"unsupported thinking protocol: {protocol!r}")


def preserve_reasoning_content(assistant: dict[str, Any], raw: dict[str, Any]) -> None:
    """Keep ModelArk's reasoning state in the next multi-turn request."""
    reasoning_content = raw.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content:
        assistant["reasoning_content"] = reasoning_content


class RetryableToolArguments(ValueError):
    """A provider emitted a malformed or cut function-call JSON string."""
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminalbench-root", required=True, type=Path)
    parser.add_argument(
        "--tasks-root",
        type=Path,
        help=(
            "Directory containing TerminalBench-compatible task directories. "
            "Defaults to <terminalbench-root>/original-tasks; the SWE-bench "
            "adapter supplies this explicitly."
        ),
    )
    parser.add_argument("--teacher", choices=("GPT", "Kimi"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--proxy-base-url", required=True)
    parser.add_argument("--proxy-model", default="agentwm-watermarked-teacher")
    parser.add_argument("--account-id", default="terminalbench-agentwm")
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--container-start-workers", type=int, default=1,
        help=(
            "Maximum simultaneous Docker compose build/pull/start operations. "
            "Agent replay stays concurrent after its container is ready; a low value prevents registry 429 bursts."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--task-profile", choices=("all", "complex"), default="all",
        help=(
            "`complex` prioritises hard tasks and longer medium tasks, then "
            "uses easy tasks as ordinary fallback candidates."
        ),
    )
    parser.add_argument(
        "--task-order-file",
        type=Path,
        help="Optional newline-delimited task-id priority order, used before the normal profile ordering.",
    )
    parser.add_argument(
        "--min-tool-calls", type=int, default=1,
        help="Silently skip an initial text-only answer instead of accepting it as an agent trace.",
    )
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high"), default="none",
        help="Optional provider reasoning effort for the teacher's normal agent turns.",
    )
    parser.add_argument(
        "--thinking-protocol", choices=("modelark", "openrouter"), default="modelark",
        help="Provider-native wire format for deep-thinking controls.",
    )
    parser.add_argument(
        "--max-tool-call-tokens", type=int, default=4096,
        help="Escalation ceiling used only after a truncated write_file call.",
    )
    parser.add_argument("--task-retries", type=int, default=3)
    parser.add_argument("--command-timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.teacher:
        args.output_dir = args.output_dir or args.output_root / "SWEbench" / "trace" / args.teacher / "_collection_work"
    if args.output_dir is None:
        parser.error("--output-dir or --teacher is required")
    return args


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def request(base_url: str, payload: dict[str, Any], retries: int = 4, timeout: float = 360) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    # The embedding proxy deliberately takes its durable session identity from
    # this header.  Keeping it aligned with the collector's per-task session
    # prevents two similar issue prompts from overwriting one another's
    # watermark trajectory/evidence files.
    headers = {"Content-Type": "application/json"}
    if session_id := str(payload.get("session_id") or "").strip():
        headers["X-AgentWM-Session"] = session_id
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers=headers, method="POST")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:800]}")
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"request failed after {retries + 1} attempts: {last_error}") from last_error


def normalise_call(raw: dict[str, Any], index: int) -> dict[str, Any]:
    function = raw.get("function") or {}
    raw_name = str(function.get("name") or "").strip()
    name = raw_name
    # Some providers serialize the already-declared TerminalBench functions
    # using their final component.  This is a wire-format compatibility map,
    # not a tool substitution: every alias resolves to one member of TOOLS.
    name = TOOL_ALIASES.get(name, name)
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            # JSON emitted by tool-call providers occasionally contains a
            # literal Windows/path backslash.  Escape only invalid JSON
            # backslashes, preserving valid escapes and argument content.
            repaired = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", arguments)
            try:
                arguments = json.loads(repaired)
            except json.JSONDecodeError as exc:
                recovered = recover_bash_command(arguments) if name == BASH else None
                if recovered is not None:
                    arguments = recovered
                else:
                    # The model response is discarded before it reaches the
                    # trajectory.  Whether the malformed JSON was cut at the
                    # limit or simply serialized incorrectly, replaying the
                    # same decision point with a larger allowance is safe.
                    raise RetryableToolArguments(
                        f"retryable invalid tool arguments for {name!r}: {exc}; raw={arguments[:300]!r}"
                    ) from exc
    if name not in TOOL_NAMES:
        translated = legacy_search_alias_to_bash(raw_name, arguments)
        if translated is not None:
            name, arguments = BASH, translated
    if name not in TOOL_NAMES or not isinstance(arguments, dict):
        raise ValueError(f"model returned a tool absent from the TerminalBench public table: {name!r}")
    if name == BASH and not isinstance(arguments.get("command"), str):
        raise ValueError("execute_bash requires a string command")
    if name in {READ, LIST} and not isinstance(arguments.get("path"), str):
        raise ValueError(f"{name} requires a string path")
    if name == WRITE and (not isinstance(arguments.get("path"), str) or not isinstance(arguments.get("content"), str)):
        raise ValueError("write_file requires string path and content")
    if name == FINISH and not isinstance(arguments.get("message"), str):
        raise ValueError("finish requires a message")
    return {"id": str(raw.get("id") or f"call_{index}"), "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


def _json_object(text: str) -> dict[str, Any] | None:
    """Parse one complete JSON object, optionally wrapped in a Markdown fence."""
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else ""
        value = value.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_objects_in_text(text: str) -> list[dict[str, Any]]:
    """Return complete JSON objects embedded in a provider reasoning trace.

    Some reasoning providers put the selected function object after ordinary
    analysis in ``reasoning`` rather than in OpenAI's ``tool_calls`` field.
    Scan only complete JSON objects; malformed/truncated fragments are left to
    the normal retry path rather than guessed.
    """
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
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


def _unique_tool_for_arguments(arguments: dict[str, Any]) -> str | None:
    """Infer a tool only when its declared argument schema makes it unique."""
    keys = set(arguments)
    schemas = {
        BASH: ({"command"}, {"command", "timeout", "wait_time_sec"}),
        READ: ({"path"}, {"path", "start_line", "end_line"}),
        WRITE: ({"path", "content"}, {"path", "content", "append"}),
        LIST: ({"path"}, {"path", "max_depth"}),
        FINISH: ({"message"}, {"message"}),
    }
    matches = [
        name for name, (required, allowed) in schemas.items()
        if required.issubset(keys) and keys.issubset(allowed)
    ]
    return matches[0] if len(matches) == 1 else None


def _reasoning_text(raw: dict[str, Any]) -> str:
    parts = [str(raw.get("reasoning") or ""), str(raw.get("content") or "")]
    for detail in raw.get("reasoning_details") or []:
        if isinstance(detail, dict):
            parts.append(str(detail.get("text") or detail.get("content") or ""))
    return "\n".join(parts)


def recover_text_tool_calls(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover a call only when a provider's textual format is unambiguous."""
    haystack = _reasoning_text(raw)
    payloads: list[dict[str, Any]] = []
    content = raw.get("content")
    if isinstance(content, str) and content.strip():
        direct = _json_object(content)
        if direct is not None:
            payloads.append(direct)
    # The direct content object is included above; this additionally captures
    # an object after explanatory reasoning.  Use the final valid object as
    # the model's selected action, never every JSON example in its analysis.
    payloads.extend(_json_objects_in_text(haystack))
    for payload in reversed(payloads):
        for name_key, arguments_key in (("name", "arguments"), ("function_name", "arguments"), ("name", "parameters")):
            name, arguments = payload.get(name_key), payload.get(arguments_key)
            if isinstance(name, str) and isinstance(arguments, dict):
                return [{"id": "text_tool_call", "type": "function", "function": {"name": name, "arguments": arguments}}]
        name = _unique_tool_for_arguments(payload)
        if name is not None:
            return [{"id": "text_tool_call", "type": "function", "function": {"name": name, "arguments": payload}}]
    return []


class FreshTask:
    def __init__(self, task_id: str, task_dir: Path, logs: Path) -> None:
        self.task_id, self.task_dir = task_id, task_dir
        self.project = "agentwm_tb_" + uuid.uuid4().hex[:16]
        self.compose = task_dir / "docker-compose.yaml"
        if not self.compose.exists():
            self.compose = task_dir / "docker-compose.yml"
        self.runtime_compose: Path | None = None
        self.runtime_dockerfile: Path | None = None
        safe = task_id.replace(".", "_").replace("/", "_").lower()
        run_suffix = self.project.removeprefix("agentwm_tb_")
        self.env = os.environ.copy()
        self.env.update({
            # A paired harmlessness run starts the same task in both arms at
            # nearly the same time.  BuildKit cannot concurrently export two
            # builds to one local tag, so make the transient client image as
            # unique as the already-unique Compose project/container.
            "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": f"tb_eval_img_{safe}_{run_suffix}",
            "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": self.project,
            "T_BENCH_TASK_LOGS_PATH": str((logs / self.project / "logs").absolute()),
            "T_BENCH_CONTAINER_LOGS_PATH": "/logs",
            "T_BENCH_TASK_AGENT_LOGS_PATH": str((logs / self.project / "agent-logs").absolute()),
            "T_BENCH_CONTAINER_AGENT_LOGS_PATH": "/agent-logs",
            "T_BENCH_TEST_DIR": "/tests",
            "COMPOSE_BAKE": "false",
        })
        self.container = ""

    def _runtime_dockerfile(self) -> Path | None:
        """Render TerminalBench's image placeholder without touching task data."""
        source = self.task_dir / "Dockerfile"
        if not source.is_file():
            return None
        content = source.read_text(encoding="utf-8")
        if "{docker_image}" not in content:
            return None
        owner, separator, instance = self.task_id.partition("__")
        if not separator or not owner or not instance:
            raise RuntimeError(f"cannot derive SWE-bench image for {self.task_id!r}")
        image = f"swebench/sweb.eval.x86_64.{owner}_1776_{instance}:latest"
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".agentwm-Dockerfile", prefix=f".{self.project}-",
            dir=self.task_dir, delete=False, encoding="utf-8",
        )
        with handle:
            handle.write(content.replace("{docker_image}", image))
        self.runtime_dockerfile = Path(handle.name)
        return self.runtime_dockerfile

    def _runtime_compose(self) -> Path:
        """Create a same-directory Compose copy with ephemeral host ports.

        TerminalBench agents execute inside the client service and keep the
        original Compose network, so service-to-service ports are unchanged.
        Only fixed *host* bindings are replaced by Docker-assigned bindings;
        this permits multiple independent task projects to coexist.
        """
        if self.runtime_compose is not None:
            return self.runtime_compose
        value = yaml.safe_load(self.compose.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("services"), dict):
            raise RuntimeError(f"invalid compose file: {self.compose}")
        changed = False
        dockerfile = self._runtime_dockerfile()
        if dockerfile is not None:
            for service in value["services"].values():
                if isinstance(service, dict) and isinstance(service.get("build"), dict):
                    # The generated compose file is in the task directory;
                    # use a basename to preserve the original build context.
                    service["build"]["dockerfile"] = dockerfile.name
                    changed = True
        for service in value["services"].values():
            if not isinstance(service, dict) or not isinstance(service.get("ports"), list):
                continue
            rewritten: list[Any] = []
            for port in service["ports"]:
                if isinstance(port, str):
                    # A single container-port entry is already dynamically
                    # published.  Any host:container form becomes an
                    # equivalent localhost dynamic-host binding.
                    pieces = port.rsplit(":", 1)
                    if len(pieces) == 2:
                        target = pieces[-1]
                        rewritten.append(f"127.0.0.1::{target}")
                        changed = True
                    else:
                        rewritten.append(port)
                elif isinstance(port, dict):
                    item = dict(port)
                    if "published" in item:
                        item.pop("published")
                        item["host_ip"] = "127.0.0.1"
                        changed = True
                    rewritten.append(item)
                else:
                    rewritten.append(port)
            service["ports"] = rewritten
        if not changed:
            return self.compose
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".agentwm-compose.yaml", prefix=f".{self.project}-",
            dir=self.task_dir, delete=False, encoding="utf-8",
        )
        with handle:
            yaml.safe_dump(value, handle, allow_unicode=True, sort_keys=False)
        self.runtime_compose = Path(handle.name)
        return self.runtime_compose

    def start(self) -> None:
        if not self.compose.exists():
            raise RuntimeError(f"missing compose file: {self.task_dir}")
        compose = self._runtime_compose()
        # TerminalBench task compose files can reference small auxiliary base
        # images (for example a decryptor's python:3.13-slim-bookworm) that
        # are not part of the task image build.  Pull only when such an image
        # is absent; never-pull makes otherwise valid tasks fail before the
        # agent sees a tool call.
        result = subprocess.run(["docker", "compose", "-f", str(compose), "-p", self.project,
                                 "up", "-d", "--pull", "missing", "--build"], capture_output=True, text=True,
                                env=self.env, timeout=900)
        detail = result.stdout + result.stderr
        if result.returncode and "Failed to fetch" in detail and "404 Not Found" in detail:
            # Some task Dockerfiles split apt-get update/install across
            # layers.  A cached update can reference packages Ubuntu has
            # already rotated away; rebuild just this task without cache.
            rebuilt = subprocess.run(["docker", "compose", "-f", str(compose), "-p", self.project,
                                      "build", "--no-cache"], capture_output=True, text=True,
                                     env=self.env, timeout=1800)
            if rebuilt.returncode:
                raise RuntimeError(f"container no-cache rebuild failed: {(rebuilt.stdout + rebuilt.stderr)[-1200:]}")
            result = subprocess.run(["docker", "compose", "-f", str(compose), "-p", self.project,
                                     "up", "-d", "--pull", "missing"], capture_output=True, text=True,
                                    env=self.env, timeout=900)
        if result.returncode:
            raise RuntimeError(f"container start failed: {(result.stdout + result.stderr)[-1200:]}")
        result = subprocess.run(["docker", "compose", "-f", str(compose), "-p", self.project, "ps", "-q"],
                                capture_output=True, text=True, env=self.env, timeout=45)
        self.container = next((line for line in result.stdout.splitlines() if line), "")
        if not self.container:
            raise RuntimeError("container did not start")

    def _run(self, command: list[str], timeout: int) -> str:
        result = subprocess.run(command, capture_output=True, text=False, timeout=timeout)

        def render(stream: bytes) -> str:
            if not stream:
                return ""
            try:
                return stream.decode("utf-8")
            except UnicodeDecodeError:
                # Tool observations are context for the teacher, not a binary
                # transport.  Preserve that a binary file was read without
                # crashing the collection or injecting replacement garbage.
                return "[binary output omitted: bytes=%d sha256=%s]" % (
                    len(stream), hashlib.sha256(stream).hexdigest()[:16]
                )

        stdout, stderr = render(result.stdout), render(result.stderr)
        text = (stdout + ("\n" + stderr if stderr else "")).strip()
        if len(text) > 16000:
            text = text[:16000] + "\n[output truncated]"
        return json.dumps({"exit_code": result.returncode, "output": text or "(no output)"}, ensure_ascii=False)

    def execute(self, name: str, arguments: dict[str, Any], default_timeout: int) -> str:
        """Execute the public TerminalBench tool without changing its semantics."""
        if name == FINISH:
            return json.dumps({"exit_code": 0, "output": str(arguments["message"])}, ensure_ascii=False)
        if name == BASH:
            raw_timeout = arguments.get("timeout")
            timeout = default_timeout if raw_timeout is None else min(default_timeout, max(1, int(float(raw_timeout))))
            return self._run(["docker", "exec", self.container, "bash", "-lc", str(arguments["command"])], timeout)
        if name == READ:
            start, end = arguments.get("start_line"), arguments.get("end_line")
            if isinstance(start, int) and isinstance(end, int):
                command = 'sed -n "${2},${3}p" -- "$1"'
                argv = ["docker", "exec", self.container, "bash", "-lc", command, "bash", str(arguments["path"]), str(start), str(end)]
            else:
                argv = ["docker", "exec", self.container, "cat", "--", str(arguments["path"])]
            return self._run(argv, default_timeout)
        if name == WRITE:
            append = "1" if arguments.get("append", False) else "0"
            command = 'if [ "$3" = 1 ]; then printf "%s" "$2" >> "$1"; else printf "%s" "$2" > "$1"; fi'
            return self._run(["docker", "exec", self.container, "bash", "-lc", command, "bash", str(arguments["path"]), str(arguments["content"]), append], default_timeout)
        if name == LIST:
            depth = int(arguments.get("max_depth", 2))
            command = 'find "$1" -maxdepth "$2" -not -name ".*" | head -100'
            return self._run(["docker", "exec", self.container, "bash", "-lc", command, "bash", str(arguments["path"]), str(depth)], default_timeout)
        raise ValueError(f"unhandled public TerminalBench tool: {name}")

    def close(self) -> None:
        compose = self.runtime_compose or self.compose
        if compose.exists():
            subprocess.run(["docker", "compose", "-f", str(compose), "-p", self.project,
                            "down", "--volumes", "--remove-orphans"], capture_output=True, text=True,
                           env=self.env, timeout=180)
        if self.runtime_compose is not None:
            self.runtime_compose.unlink(missing_ok=True)
            self.runtime_compose = None
        if self.runtime_dockerfile is not None:
            self.runtime_dockerfile.unlink(missing_ok=True)
            self.runtime_dockerfile = None


def task_instruction(path: Path) -> str:
    value = yaml.safe_load((path / "task.yaml").read_text(encoding="utf-8"))
    instruction = value.get("instruction") if isinstance(value, dict) else None
    if not isinstance(instruction, str) or not instruction.strip():
        raise RuntimeError(f"task has no instruction: {path.name}")
    return instruction.strip()


def task_item(path: Path) -> dict[str, Any]:
    """Read public task metadata used solely to order collection candidates."""
    value = yaml.safe_load((path / "task.yaml").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid task metadata: {path.name}")
    return {
        "run_id": path.name,
        "source_task_id": path.name,
        "task_path": str(path),
        "difficulty": str(value.get("difficulty") or "").lower(),
        "category": str(value.get("category") or ""),
        "expert_time_estimate_min": int(value.get("expert_time_estimate_min") or 0),
        "junior_time_estimate_min": int(value.get("junior_time_estimate_min") or 0),
        "estimated_duration_sec": int(value.get("estimated_duration_sec") or 0),
    }


def ordered_items(paths: list[Path], profile: str) -> list[dict[str, Any]]:
    items = [task_item(path) for path in paths]
    rank = {"hard": 2, "medium": 1, "easy": 0}
    if profile == "all":
        rank = {"hard": 1, "medium": 1, "easy": 1}
    return sorted(
        items,
        key=lambda item: (
            -rank.get(item["difficulty"], -1),
            -item["expert_time_estimate_min"],
            -item["junior_time_estimate_min"],
            -item["estimated_duration_sec"],
            item["run_id"],
        ),
    )


def collect_one(
    args: argparse.Namespace,
    item: dict[str, str],
    container_start_gate: threading.BoundedSemaphore,
) -> dict[str, Any]:
    task = FreshTask(item["source_task_id"], Path(item["task_path"]), args.output_dir / "collection" / "container_logs")
    session_id = "swebench-" + item["run_id"] + "-" + str(time.time_ns())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt_with_tools()},
        {"role": "user", "content": task_instruction(Path(item["task_path"]))},
    ]
    steps = 0
    try:
        # Docker Hub rate-limits concurrent image manifest/pull requests far
        # more aggressively than model inference.  Gate only this setup
        # phase: once a task container is live, its teacher/tool loop releases
        # the gate and can run concurrently with other ready tasks.
        with container_start_gate:
            task.start()
        generation_tokens = args.max_tokens
        while steps < args.max_steps:
            payload = {"model": args.proxy_model, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
                       "temperature": 0, "max_tokens": generation_tokens, "session_id": session_id,
                       "user": args.account_id, "benchmark": "swebench", "domain": "terminal",
                       "task_id": item["run_id"]}
            add_reasoning_config(payload, args.reasoning_effort, args.thinking_protocol)
            raw = request(args.proxy_base_url, payload).get("choices", [{}])[0].get("message") or {}
            calls = raw.get("tool_calls") or []
            recovered_text_call = False
            if not calls:
                calls = recover_text_tool_calls(raw)
                recovered_text_call = bool(calls)
            # A recovered bare JSON argument object is serialized below as a
            # canonical tool call.  Keeping it as assistant text as well
            # would teach an accidental provider wire format.
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": None if recovered_text_call else raw.get("content"),
            }
            preserve_reasoning_content(assistant, raw)
            if not calls:
                messages.append(assistant)
                return {"schema": "agentwm_swebench_collection_trace_v1", "task_id": item["run_id"],
                        "source_task_id": item["source_task_id"], "session_id": payload["session_id"],
                        "messages": messages, "tool_schemas": TOOLS, "steps": steps,
                        "termination_reason": "no_tool_calls" if steps == 0 else "completed"}
            try:
                normalised = [normalise_call(call, steps * 10 + index) for index, call in enumerate(calls)]
            except RetryableToolArguments:
                if generation_tokens >= args.max_tool_call_tokens:
                    raise
                # 384 -> 512 first, then only grow when the larger retry is
                # still insufficient.  No invalid message is appended, so
                # this replays precisely the same decision point.
                generation_tokens = min(
                    args.max_tool_call_tokens,
                    512 if generation_tokens < 512 else generation_tokens * 2,
                )
                continue
            assistant["tool_calls"] = normalised
            messages.append(assistant)
            for call in normalised:
                function = call["function"]
                arguments = json.loads(function["arguments"])
                try:
                    observation = task.execute(str(function["name"]), arguments, args.command_timeout)
                except subprocess.TimeoutExpired:
                    observation = json.dumps({"exit_code": 124, "output": f"tool timed out after {args.command_timeout}s"})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
            steps += 1
            generation_tokens = args.max_tokens
            if any(call["function"]["name"] == FINISH for call in normalised):
                return {"schema": "agentwm_swebench_collection_trace_v1", "task_id": item["run_id"],
                        "source_task_id": item["source_task_id"], "session_id": payload["session_id"],
                        "messages": messages, "tool_schemas": TOOLS, "steps": steps,
                        "termination_reason": "finished"}
        return {"schema": "agentwm_swebench_collection_trace_v1", "task_id": item["run_id"],
                "source_task_id": item["source_task_id"], "session_id": "swebench-" + item["run_id"],
                "messages": messages, "tool_schemas": TOOLS, "steps": steps,
                "termination_reason": "max_steps"}
    finally:
        task.close()


def collect(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.target_count < 1 or args.workers < 1 or args.container_start_workers < 1:
        raise ValueError("target count, workers, and container-start-workers must be positive")
    if args.container_start_workers > args.workers:
        raise ValueError("container-start-workers cannot exceed workers")
    task_root = args.tasks_root or (args.terminalbench_root / "original-tasks")
    task_root = task_root.resolve()
    if not task_root.is_dir():
        raise RuntimeError(f"task root does not exist: {task_root}")
    task_paths = [path for path in task_root.iterdir() if path.is_dir() and (path / "task.yaml").is_file() and ((path / "docker-compose.yaml").is_file() or (path / "docker-compose.yml").is_file())]
    if args.task_order_file:
        requested = [
            line.strip()
            for line in args.task_order_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {path.name: path for path in task_paths}
        missing = [task_id for task_id in requested if task_id not in by_id]
        if missing:
            raise RuntimeError(f"ordered task ids are missing from {task_root}: {missing[:3]}")
        # Preserve the auditable benchmark-selection order exactly.  Remaining
        # task directories are never reached for a bounded target unless an
        # earlier candidate is rejected.
        seen: set[str] = set()
        items = []
        for task_id in requested:
            if task_id not in seen:
                items.append(task_item(by_id[task_id]))
                seen.add(task_id)
        items.extend(item for item in ordered_items(task_paths, args.task_profile) if item["run_id"] not in seen)
    else:
        items = ordered_items(task_paths, args.task_profile)
    if len(items) < args.target_count:
        raise RuntimeError(f"only {len(items)} runnable unique tasks for profile {args.task_profile!r}, target is {args.target_count}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "collection" / "candidate_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "collection" / "candidate_manifest.json").write_text(json.dumps({
        "target": args.target_count, "task_profile": args.task_profile, "candidates": items,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    accepted_path = args.output_dir / "collection" / "accepted.jsonl"
    accepted = {json.loads(line)["task_id"] for line in accepted_path.read_text(encoding="utf-8").splitlines()} if args.resume and accepted_path.exists() else set()
    trace_dir = args.output_dir / "collection" / "traces"; trace_dir.mkdir(parents=True, exist_ok=True)
    by_id = {item["run_id"]: item for item in items}
    retry_first: list[dict[str, str]] = []
    retry_ids: set[str] = set()
    rejected_path = args.output_dir / "collection" / "rejected.jsonl"
    if args.resume and rejected_path.exists():
        seen_retry_ids: set[str] = set()
        for line in rejected_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                task_id = str(row.get("task_id") or "")
            except json.JSONDecodeError:
                continue
            if task_id in by_id and task_id not in accepted and task_id not in seen_retry_ids:
                retry_first.append(by_id[task_id])
                seen_retry_ids.add(task_id)
                retry_ids.add(task_id)
        # The failures above are now held in the in-memory priority queue for
        # this resumed run.  Remove the stale report so collection state shows
        # only failures that remain after this retry pass.
        rejected_path.unlink()
    pending = retry_first + [item for item in items if item["run_id"] not in accepted and item["run_id"] not in retry_ids]
    iterator = iter(pending); active: dict[concurrent.futures.Future[dict[str, Any]], dict[str, str]] = {}

    def progress(status: str) -> None:
        progress_path = args.output_dir / "collection" / "progress.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps({"status": status, "accepted": len(accepted), "target": args.target_count, "active": len(active), "available_unique_tasks": len(items), "task_profile": args.task_profile}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    container_start_gate = threading.BoundedSemaphore(args.container_start_workers)

    def run_retry(item: dict[str, str]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(args.task_retries + 1):
            try:
                return collect_one(args, item, container_start_gate)
            except Exception as exc:
                last = exc
                if attempt < args.task_retries:
                    detail = str(exc).lower()
                    # A registry 429 is shared state, so retrying the same
                    # request a few seconds later only prolongs the outage.
                    # Back off long enough for the manifest-pull window to
                    # cool down; other already-ready agent loops still run.
                    base_delay = 30 if "429" in detail or "too many requests" in detail else 2
                    time.sleep(min(180, base_delay * (2 ** attempt)))
        raise RuntimeError(f"failed after {args.task_retries + 1} attempts: {last}")

    progress("running")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(accepted) < args.target_count:
            while len(active) < args.workers and len(accepted) + len(active) < args.target_count:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                active[pool.submit(run_retry, item)] = item
            # Make progress reflect submitted, still-running task containers
            # immediately; otherwise a slow first batch appears as active=0
            # until its first completion.
            progress("running")
            if not active:
                break
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                item = active.pop(future)
                try:
                    trace = future.result()
                    calls = sum(
                        len(message.get("tool_calls") or [])
                        for message in trace.get("messages", [])
                        if isinstance(message, dict) and message.get("role") == "assistant"
                    )
                    if calls < args.min_tool_calls:
                        # Decide only after the complete trace has been
                        # assembled.  A multi-turn adapter may legitimately
                        # have an early text reply followed by a Core action.
                        # A trace with no Core action at all is silently
                        # skipped and the next candidate fills this slot.
                        continue
                    path = trace_dir / f"{item['run_id']}.json"
                    # Keep an accepted trace durable if a previous failed run
                    # was cleaned while an old worker was still unwinding.
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    accepted.add(item["run_id"])
                    append_jsonl(accepted_path, {"task_id": item["run_id"], "source_task_id": item["source_task_id"], "session_id": trace["session_id"], "termination_reason": trace["termination_reason"], "trace": str(path)})
                except Exception as exc:
                    append_jsonl(args.output_dir / "collection" / "rejected.jsonl", {"task_id": item["run_id"], "source_task_id": item["source_task_id"], "error": str(exc)})
                progress("running")
    progress("complete" if len(accepted) >= args.target_count else "exhausted")
    if len(accepted) < args.target_count:
        raise RuntimeError(f"only collected {len(accepted)}/{args.target_count} traces")


def main() -> None:
    """Single public entrypoint used by SWE-bench shell launchers.

    Collection retries each task locally, then rejects it and keeps consuming
    the ordered candidate pool until the exact requested accepted count is
    reached.  A later ``--resume`` pass retries persisted rejections first.
    """
    argv = sys.argv[1:]
    if not argv or argv[0] == "collect":
        collect(argv[1:] if argv else None)
        return
    command, rest = argv[0], argv[1:]
    if command == "prepare-candidates":
        _prepare_candidates(rest)
        return
    if command == "freeze":
        _freeze_selection(rest)
        return
    if command == "robustness":
        from robustness import run_benchmark
        run_benchmark("swebench", rest)
        return
    raise SystemExit(
        f"unknown SWE-bench command {command!r}; "
        "use prepare-candidates, collect, freeze, or robustness"
    )


def _freeze_selection(argv: list[str] | None = None) -> None:
    """Freeze accepted S/D_c traces and matching watermark evidence."""
    import argparse
    import json
    import shutil
    from pathlib import Path
    from typing import Any


    def jsonl(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


    def message_quality(messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Recompute tool-call pairing from an executable collector trace."""
        call_ids = [
            str(call.get("id") or "")
            for message in messages if isinstance(message, dict)
            for call in (message.get("tool_calls") or []) if isinstance(call, dict) and call.get("id")
        ]
        observed = {
            str(message.get("tool_call_id") or "")
            for message in messages if isinstance(message, dict) and message.get("role") == "tool"
        }
        paired = sum(call_id in observed for call_id in call_ids)
        return {
            "assistant_turns": sum(message.get("role") == "assistant" for message in messages if isinstance(message, dict)),
            "tool_calls": len(call_ids),
            "paired_tool_calls": paired,
            "unpaired_tool_calls": len(call_ids) - paired,
            "trainable": bool(call_ids) and paired == len(call_ids),
        }


    def collector_complete_record(run_dir: Path, task_id: str, row: dict[str, Any]) -> dict[str, Any]:
        """Use the collector's complete prefix when the proxy saw a final call.

    The proxy records its request before the collector later appends the final
    tool observation.  The collector trace is the same interaction plus that
    observation, so using it here preserves a fully paired max-step prefix
    without modifying the watermark endpoint.
    """
        path = run_dir / "collection" / "traces" / f"{task_id}.json"
        try:
            collected = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return row
        if collected.get("schema") not in {
            "agentwm_swebench_collection_trace_v1",
            "agentwm_travelplanner_trace_v1",
            "agentwm_tau2_collection_trace_v1",
        }:
            return row
        messages, tools = collected.get("messages"), collected.get("tool_schemas")
        if not isinstance(messages, list) or not isinstance(tools, list):
            return row
        value = dict(row)
        value["messages"], value["tool_schemas"] = messages, tools
        value["quality"] = message_quality(messages)
        value["termination_reason"] = collected.get("termination_reason")
        return value


    def main(argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--bench", choices=("BFCL", "SWEbench", "Tau2"))
        parser.add_argument("--teacher", choices=("GPT", "Kimi"))
        parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "output")
        parser.add_argument("--run-dir", type=Path)
        parser.add_argument("--out-dir", type=Path)
        parser.add_argument("--count", type=int, default=50)
        parser.add_argument(
            "--seed", type=int, default=42,
            help="Retained for launcher compatibility; S/D_c use collection order rather than random sampling.",
        )
        parser.add_argument(
            "--require-aux-release",
            action="store_true",
            help="Select only trajectories whose paired evidence records at least one aux_release event.",
        )
        parser.add_argument(
            "--write-back-complete",
            action="store_true",
            help="Synchronise collector-complete messages back into accepted canonical standard trajectories before freezing.",
        )
        args = parser.parse_args(argv)
        if args.bench:
            if not args.teacher:
                parser.error("--bench requires --teacher")
            trace_root = args.output_root / args.bench / "trace" / args.teacher
            args.run_dir = args.run_dir or trace_root / "_collection_work"
            args.out_dir = args.out_dir or trace_root / ("S" if args.count == 100 else "D_c" if args.count == 50 else f"selection_{args.count}")
        if args.run_dir is None or args.out_dir is None:
            parser.error("--run-dir and --out-dir are required without --bench")
        accepted_rows = jsonl(args.run_dir / "collection" / "accepted.jsonl")
        accepted_order = [str(row["task_id"]) for row in accepted_rows]
        accepted_sessions = {str(row["task_id"]): str(row.get("session_id") or "") for row in accepted_rows}
        accepted = set(accepted_order)
        records: dict[str, tuple[Path, dict[str, Any]]] = {}
        source = args.run_dir / "embedding" / "standard_traces"
        for path in source.glob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            task_id = str(row.get("task_id") or "")
            if task_id not in accepted or row.get("trace_kind") != "watermarked":
                continue
            # A task may have failed/retried proxy sessions before the collector
            # accepted its final session.  Never let a later retry snapshot replace
            # the accepted trace solely because it has a newer mtime.
            if accepted_sessions.get(task_id) and str(row.get("session_id") or "") != accepted_sessions[task_id]:
                continue
            original_row = row
            row = collector_complete_record(args.run_dir, task_id, row)
            if args.write_back_complete and row is not original_row:
                # The collector trace is the same watermarked dialogue plus any
                # final tool observation that arrived after the proxy's request
                # snapshot.  Make the canonical standard trajectory complete so
                # all later consumers see identical, fully paired history.
                path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            # Collection termination is deliberately irrelevant here: both a
            # completed trajectory and a max_steps prefix are normal accepted
            # traces.  D_c selection is governed only by whether the recorded
            # watermark artifact itself is trainable.
            if not row.get("quality", {}).get("trainable"):
                continue
            if args.require_aux_release:
                evidence = args.run_dir / "embedding" / "watermark_evidence" / path.name
                try:
                    events = json.loads(evidence.read_text(encoding="utf-8")).get("events", [])
                except (OSError, json.JSONDecodeError):
                    continue
                if not any(isinstance(event, dict) and event.get("event") == "aux_release" for event in events):
                    continue
            if task_id not in records or path.stat().st_mtime > records[task_id][0].stat().st_mtime:
                records[task_id] = (path, row)
        eligible = [task_id for task_id in accepted_order if task_id in records]
        if len(eligible) < args.count:
            raise RuntimeError(f"only {len(eligible)}/{args.count} accepted trainable records are available")
        # D_c is the first ``count`` complete collected traces in durable collector
        # order.  ``--require-aux-release`` optionally narrows the eligible
        # population, but is deliberately not the default: non-release context is
        # also part of a faithful distillation corpus.
        selected = eligible[:args.count]
        for directory in (args.out_dir, args.out_dir / "standard_traces", args.out_dir / "watermark_evidence", args.out_dir / "watermarked_traces"):
            directory.mkdir(parents=True, exist_ok=True)
        output = []
        manifest = []
        for task_id in selected:
            path, row = records[task_id]
            (args.out_dir / "standard_traces" / path.name).write_text(
                json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            evidence = args.run_dir / "embedding" / "watermark_evidence" / path.name
            if not evidence.exists():
                raise RuntimeError(f"missing watermark evidence for {task_id}: {evidence.name}")
            shutil.copyfile(evidence, args.out_dir / "watermark_evidence" / evidence.name)
            visible = args.run_dir / "embedding" / "watermarked_traces" / path.name
            if not visible.exists():
                raise RuntimeError(f"missing watermarked trace for {task_id}: {visible.name}")
            shutil.copyfile(visible, args.out_dir / "watermarked_traces" / visible.name)
            output.append(row)
            manifest.append({"task_id": task_id, "trace_id": row.get("trace_id"), "artifact": path.name})
        (args.out_dir / "train.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
        (args.out_dir / "manifest.json").write_text(json.dumps({"count": len(manifest), "selection": "collection_order_first_n", "tasks": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"count": len(manifest), "selection": "collection_order_first_n", "out_dir": str(args.out_dir)}, ensure_ascii=False))

    main(argv)


def _prepare_candidates(argv: list[str] | None = None) -> None:
    """Prepare the SWEbench candidate pool from official metadata."""
    import argparse
    import hashlib
    import json
    import re
    import sys
    from pathlib import Path
    from typing import Any

    import pyarrow.parquet as pq


    def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--parquet", type=Path, required=True)
        parser.add_argument("--adapter-dir", type=Path, required=True)
        parser.add_argument("--swebench-root", type=Path, required=True)
        parser.add_argument("--out-dir", type=Path, required=True)
        parser.add_argument(
            "--target-count", type=int, default=100,
            help="Number of final accepted teacher traces requested from the later collector.",
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--candidate-count", type=int, default=300,
            help="Materialised gold-patch-complex candidate tasks; must exceed target-count.",
        )
        parser.add_argument(
            "--selection-mode",
            choices=("complex", "uniform", "simple"),
            default="complex",
            help=(
                "`complex` keeps the original gold-patch-complex ranking. "
                "`uniform` takes a deterministic uniform ordering of every row; "
                "use it for a fixed SWE-bench Verified evaluation subset. "
                "`simple` prioritises bounded, single-file production fixes for "
                "a lightweight outcome-evaluation subset."
            ),
        )
        parser.add_argument(
            "--dataset-label",
            default="princeton-nlp/SWE-bench",
            help="Provenance label recorded in task_selection.json.",
        )
        parser.add_argument(
            "--tiny-single-file-lines", type=int, default=5,
            help="Reject a one-production-file/one-hunk patch at or below this many changed lines.",
        )
        return parser.parse_args(argv)


    def list_size(value: Any) -> int:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return 0
        return len(value) if isinstance(value, list) else 0


    def _patch_path(line: str) -> str | None:
        """Extract a destination path from a unified-diff file header."""
        if not line.startswith("+++ "):
            return None
        path = line[4:].strip().split("\t", 1)[0]
        if path == "/dev/null":
            return None
        return path.removeprefix("a/").removeprefix("b/")


    def _is_test_or_nonproduction(path: str) -> bool:
        lower = path.lower().replace("\\", "/")
        parts = lower.split("/")
        if any(part in {"test", "tests", "testing", "docs", "doc", "examples", "example"} for part in parts):
            return True
        filename = parts[-1]
        if filename.startswith("test_") or filename.endswith("_test.py"):
            return True
        return filename.endswith((".md", ".rst", ".txt")) or "changelog" in lower


    def patch_stats(patch: Any) -> dict[str, int]:
        """Count non-test gold-patch work without retaining patch content."""
        current: str | None = None
        per_file: dict[str, dict[str, int]] = {}
        for line in str(patch or "").splitlines():
            path = _patch_path(line)
            if path is not None:
                current = path
                per_file.setdefault(current, {"hunks": 0, "changed_lines": 0})
                continue
            if current is None:
                continue
            if line.startswith("@@"):
                per_file[current]["hunks"] += 1
            elif (line.startswith("+") and not line.startswith("+++")) or (
                line.startswith("-") and not line.startswith("---")
            ):
                per_file[current]["changed_lines"] += 1

        production = {path: value for path, value in per_file.items() if not _is_test_or_nonproduction(path)}
        return {
            "changed_files": len(per_file),
            "production_files": len(production),
            "test_or_aux_files": len(per_file) - len(production),
            "production_hunks": sum(value["hunks"] for value in production.values()),
            "production_changed_lines": sum(value["changed_lines"] for value in production.values()),
        }


    def eligible(stats: dict[str, int], tiny_single_file_lines: int) -> bool:
        """Exclude test-only patches and objectively tiny one-line fixes."""
        if stats["production_files"] == 0 or stats["production_hunks"] == 0:
            return False
        return not (
            stats["production_files"] == 1
            and stats["production_hunks"] == 1
            and stats["production_changed_lines"] <= tiny_single_file_lines
        )


    def priority(row: dict[str, Any], stats: dict[str, int], seed: int) -> tuple[int, int, int, int, int, str]:
        """Gold-patch complexity first; test counts are only a tie-breaking signal."""
        failing = list_size(row.get("FAIL_TO_PASS"))
        # Cap each component so a generated/giant diff cannot monopolise the pool.
        tie = hashlib.sha256(f"{seed}:{row['instance_id']}".encode()).hexdigest()
        return (
            -min(stats["production_files"], 8),
            -min(stats["production_hunks"], 16),
            -min(stats["production_changed_lines"], 200),
            -min(stats["changed_files"], 12),
            -min(failing, 12),
            tie,
        )


    def uniform_priority(row: dict[str, Any], seed: int) -> str:
        """Stable, answer-independent permutation for a benchmark subset."""
        return hashlib.sha256(f"{seed}:{row['instance_id']}".encode()).hexdigest()


    def simple_priority(row: dict[str, Any], stats: dict[str, int], seed: int) -> tuple[int, int, int, int, str]:
        """Prefer nontrivial but small production fixes without exposing patches.

    ``eligible`` has already removed test-only and one-line trivial patches.
    The remaining ranking favours one-file/one-hunk changes with at most a
    few dozen production lines, a practical proxy for tasks an autonomous
    terminal agent can complete in a bounded harmlessness evaluation.
    """
        tie = hashlib.sha256(f"{seed}:{row['instance_id']}".encode()).hexdigest()
        return (
            stats["production_files"],
            stats["production_hunks"],
            stats["production_changed_lines"],
            stats["changed_files"],
            tie,
        )


    def summary(row: dict[str, Any], stats: dict[str, int], rank: int) -> dict[str, Any]:
        return {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "selection_rank": rank,
            "gold_patch_complexity": stats,
            "fail_to_pass_tests": list_size(row.get("FAIL_TO_PASS")),
            "pass_to_pass_tests": list_size(row.get("PASS_TO_PASS")),
        }


    def main(argv: list[str] | None = None) -> None:
        args = parse_args(argv)
        if args.target_count < 1 or args.candidate_count < args.target_count:
            raise ValueError("candidate-count must be at least target-count and both must be positive")
        rows = pq.read_table(args.parquet).to_pylist()
        annotated = [
            (row, patch_stats(row.get("patch")))
            for row in rows
        ]
        eligible_rows = [
            (row, stats)
            for row, stats in annotated
            if eligible(stats, args.tiny_single_file_lines)
        ]
        if args.selection_mode == "complex":
            ranked = sorted(eligible_rows, key=lambda value: priority(value[0], value[1], args.seed))
            selection_pool = len(eligible_rows)
            selection_description = (
                "gold-patch complexity: prioritise non-test production files, hunks, and changed lines; "
                "FAIL_TO_PASS is a capped tie-breaker only; exclude test-only and tiny single-file patches"
            )
        elif args.selection_mode == "uniform":
            # Verified already filters for human-confirmed, solvable tasks.  Do not
            # use patch complexity a second time, which would reintroduce a hard
            # task bias into the harmlessness comparison.
            ranked = sorted(annotated, key=lambda value: uniform_priority(value[0], args.seed))
            selection_pool = len(annotated)
            selection_description = "deterministic uniform permutation over all published rows; no gold-patch ranking"
        else:
            ranked = sorted(eligible_rows, key=lambda value: simple_priority(value[0], value[1], args.seed))
            selection_pool = len(eligible_rows)
            selection_description = (
                "small nontrivial production fixes from SWE-bench Full: rank ascending by production files, "
                "hunks, changed lines, and total files; test-only and <=5-line single-file fixes excluded"
            )
        if args.candidate_count > len(ranked):
            raise ValueError(f"candidate-count {args.candidate_count} exceeds selectable rows {len(ranked)}")

        # The adapter is vendor code with a local-record extension.  Put both its
        # directory and the official SWE-bench checkout on the import path.
        sys.path.insert(0, str(args.adapter_dir))
        sys.path.insert(0, str(args.swebench_root))
        from adapter import SWEBenchAdapter  # type: ignore[import-not-found]

        task_dir = args.out_dir / "tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        # Give the adapter its entire eligible lookup table.  The candidate order,
        # rather than a gold patch, decides what is materialised and collected.
        adapter = SWEBenchAdapter(task_dir=task_dir, records=[row for row, _ in ranked])
        prepared: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for row, stats in ranked:
            if len(prepared) >= args.candidate_count:
                break
            instance_id = str(row["instance_id"])
            try:
                adapter.generate_task(instance_id, instance_id)
            except Exception as exc:
                failures.append({"instance_id": instance_id, "error": f"{type(exc).__name__}: {exc}"})
                continue
            prepared.append(summary(row, stats, len(prepared) + 1))
        if len(prepared) != args.candidate_count:
            raise RuntimeError(f"only prepared {len(prepared)}/{args.candidate_count} candidates; failures={len(failures)}")

        order_path = args.out_dir / "task_priority_order.txt"
        order_path.write_text("\n".join(item["instance_id"] for item in prepared) + "\n", encoding="utf-8")
        (args.out_dir / "task_selection.json").write_text(
            json.dumps(
                {
                    "dataset": args.dataset_label,
                    "split": "test",
                    "selection": selection_description,
                    "selection_mode": args.selection_mode,
                    "gold_patch_used_for_selection": args.selection_mode in {"complex", "simple"},
                    "gold_patch_exposed_to_teacher_prompt": False,
                    "final_trace_target_count": args.target_count,
                    "candidate_count": args.candidate_count,
                    "eligible_pool_count": selection_pool,
                    "tiny_single_file_lines": args.tiny_single_file_lines,
                    "candidates": prepared,
                    "conversion_failures": failures,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"prepared {len(prepared)} candidate tasks under {task_dir}", flush=True)

    main(argv)


if __name__ == "__main__":
    main()
