#!/usr/bin/env python3
"""Tau2 text adapter: real S collection, S/D_c freezing, and D mixtures."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def prepare_tau2(root: Path) -> None:
    root = root.resolve()
    if not (root / "src" / "tau2").is_dir() or not (root / "data").is_dir():
        raise RuntimeError(f"not a Tau2 checkout: {root}")
    os.environ.setdefault("TAU2_DATA_DIR", str(root / "data"))
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def proxy_request(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-AgentWM-Session": str(payload["session_id"]),
        "X-AgentWM-Account": str(payload["user"]),
    }
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, headers=headers, method="POST")
    last: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=360) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:800]}")
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise last from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
        if attempt < 3:
            time.sleep(2**attempt)
    raise RuntimeError(f"proxy request failed: {last}") from last


def openai_messages(messages: list[Any]) -> list[dict[str, Any]]:
    from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage
    output: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, (SystemMessage, UserMessage)):
            output.append({"role": message.role, "content": message.content or ""})
        elif isinstance(message, AssistantMessage):
            row: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                row["tool_calls"] = [{"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}} for call in message.tool_calls]
            output.append(row)
        elif isinstance(message, ToolMessage) and message.requestor == "assistant":
            output.append({"role": "tool", "tool_call_id": message.id, "content": message.content or ""})
    return output


def make_agent(*, tools: list[Any], domain_policy: str, base_url: str, model: str,
               account: str, session_id: str, domain: str, task_id: str, max_tokens: int) -> Any:
    """Use Tau2's LLMAgent state machine while preserving AgentWM session headers."""
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall

    class Agent(LLMAgent):
        def _generate_next_message(self, message: Any, state: Any) -> Any:
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            payload = {
                "model": self.llm,
                "messages": openai_messages([*state.system_messages, *state.messages]),
                "tools": [tool.openai_schema for tool in self.tools], "tool_choice": "auto",
                "temperature": 0.0, "max_tokens": max_tokens, "session_id": session_id,
                "user": account, "benchmark": "tau2", "domain": domain, "task_id": task_id,
            }
            raw = proxy_request(base_url, payload).get("choices", [{}])[0].get("message") or {}
            allowed = {tool.openai_schema["function"]["name"] for tool in self.tools}
            calls = []
            for index, call in enumerate(raw.get("tool_calls") or []):
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                if name not in allowed:
                    raise ValueError(f"tool absent from official Tau2 table: {name!r}")
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError(f"arguments for {name!r} are not an object")
                calls.append(ToolCall(id=str(call.get("id") or f"call_{index}"), name=name, arguments=arguments, requestor="assistant"))
            return AssistantMessage(role="assistant", content=raw.get("content"), tool_calls=calls or None)

    return Agent(tools=tools, domain_policy=domain_policy, llm=model, llm_args={})


def proxy_aux_url(base_url: str, endpoint: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root + endpoint


def post_aux_observation(base_url: str, *, session_id: str, tool_call_id: str, observation: str) -> str:
    """Tell the proxy about a terminal Aux result, if this ID is its pending Aux.

    The proxy distinguishes a terminal Core (``not_aux``) from a recorded Aux.
    An ID mismatch is never ignored: it would leave a private release unpaired.
    """
    payload = json.dumps({"session_id": session_id, "tool_call_id": tool_call_id, "observation": observation}).encode("utf-8")
    request = urllib.request.Request(proxy_aux_url(base_url, "/v1/aux-observation"), data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = str(json.loads(response.read().decode("utf-8")).get("status") or "")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"terminal Aux observation failed: HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}") from exc
    if status not in {"recorded", "not_aux"}:
        raise RuntimeError(f"terminal Aux observation returned unexpected status: {status!r}")
    return status


def verify_aux_pairing(base_url: str, *, session_id: str) -> None:
    """Require the proxy evidence stream to be fully paired before acceptance."""
    url = proxy_aux_url(base_url, "/v1/aux-status") + "?" + urllib.parse.urlencode({"session_id": session_id})
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            status = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Aux pairing audit failed: HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}") from exc
    if status.get("pending_call_id") or status.get("unpaired_release_ids"):
        raise RuntimeError(
            "proxy evidence has unpaired Aux calls: "
            f"pending={status.get('pending_call_id')!r}, releases={status.get('unpaired_release_ids')!r}"
        )


def trace_messages(agent: Any, simulation: Any, environment: Any, *, proxy_base_url: str, session_id: str) -> list[dict[str, Any]]:
    """Return a fully paired official transcript, including a terminal tool result.

    Tau2 can stop exactly after an assistant tool-call turn.  Unlike a regular
    next turn, the final environment response is then absent from its returned
    transcript.  SWE executes that returned call before accepting the bounded
    prefix; do the identical real-environment completion here.
    """
    from tau2.data_model.message import AssistantMessage
    messages = list(simulation.get_messages())
    if messages and isinstance(messages[-1], AssistantMessage) and messages[-1].tool_calls:
        for call in messages[-1].tool_calls:
            result = environment.get_response(call)
            messages.append(result)
            # ToolMessage.id is the exact ID that Tau2 bound to the assistant
            # tool call.  It is also the public ID returned by the proxy.
            post_aux_observation(proxy_base_url, session_id=session_id, tool_call_id=str(result.id), observation=result.content or "")
    # This is the actual official transcript. User-private tool traffic is not
    # part of the agent history and is intentionally filtered by openai_messages.
    from tau2.data_model.message import SystemMessage
    verify_aux_pairing(proxy_base_url, session_id=session_id)
    return openai_messages([SystemMessage(role="system", content=agent.system_prompt), *messages])


def quality(messages: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [str(call.get("id") or "") for message in messages if message.get("role") == "assistant" for call in (message.get("tool_calls") or [])]
    observations = {str(message.get("tool_call_id") or "") for message in messages if message.get("role") == "tool"}
    paired = sum(call_id in observations for call_id in calls)
    return {"assistant_turns": sum(message.get("role") == "assistant" for message in messages), "tool_calls": len(calls), "paired_tool_calls": paired, "unpaired_tool_calls": len(calls) - paired, "trainable": bool(calls) and paired == len(calls)}


def collect_one(args: argparse.Namespace, task: Any) -> dict[str, Any]:
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import build_environment, build_user
    from tau2.runner.simulation import run_simulation
    session_id = f"tau2-{args.domain}-{task.id}-{time.time_ns()}"
    environment = build_environment(args.domain)
    tools = environment.get_tools()
    agent = make_agent(tools=tools, domain_policy=environment.get_policy(), base_url=args.proxy_base_url,
                       model=args.proxy_model, account=args.account_id, session_id=session_id,
                       domain=args.domain, task_id=str(task.id), max_tokens=args.max_tokens)
    user = build_user("user_simulator", environment, task, llm=args.user_model, llm_args=args.user_llm_args)
    simulation = run_simulation(Orchestrator(domain=args.domain, agent=agent, user=user, environment=environment,
                                task=task, max_steps=args.max_steps, timeout=args.timeout, simulation_id=session_id))
    messages = trace_messages(agent, simulation, environment, proxy_base_url=args.proxy_base_url, session_id=session_id)
    q = quality(messages)
    return {
        "schema": "agentwm_tau2_collection_trace_v1", "benchmark": "tau2", "domain": args.domain,
        "task_id": str(task.id), "session_id": session_id, "messages": messages,
        "tool_schemas": [tool.openai_schema for tool in tools], "steps": q["tool_calls"],
        "termination_reason": simulation.termination_reason.value, "quality": q,
        "reward": simulation.reward_info.model_dump() if simulation.reward_info else None,
        "source": {"task_set": args.task_set, "task_split": args.task_split},
    }


def collect_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--teacher", choices=("GPT", "Kimi"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--proxy-base-url", required=True)
    parser.add_argument("--proxy-model", default="agentwm-watermarked-teacher")
    parser.add_argument("--account-id", default="tau2-agentwm")
    parser.add_argument("--domain", default="telecom")
    parser.add_argument("--task-set", default="telecom_full",
                        help="Use the full Telecom candidate pool, then freeze S by collection order.")
    parser.add_argument("--task-split", default="all",
                        help="Use all for telecom_full; named splits are supported by the telecom task set.")
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument(
        "--min-reference-actions", type=int, default=1,
        help="Keep only official tasks with at least this many reference assistant actions.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--min-tool-calls", type=int, default=1)
    parser.add_argument("--task-retries", type=int, default=3)
    parser.add_argument(
        "--selection-seed", type=int, default=42,
        help="Fixed seed for a representative full-pool task order; retries retain priority on resume.",
    )
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--user-llm-args", default='{"temperature": 0.0}')
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.teacher:
        args.output_dir = args.output_dir or args.output_root / "Tau2" / "trace" / args.teacher / "_collection_work"
    if args.output_dir is None:
        parser.error("--output-dir or --teacher is required")
    try:
        args.user_llm_args = json.loads(args.user_llm_args)
    except json.JSONDecodeError as exc:
        raise ValueError("--user-llm-args must be a JSON object") from exc
    if not isinstance(args.user_llm_args, dict):
        raise ValueError("--user-llm-args must be a JSON object")
    if min(args.target_count, args.workers, args.min_tool_calls, args.min_reference_actions) < 1:
        raise ValueError("target-count, workers, min-tool-calls, and min-reference-actions must be positive")
    return args


def collect(argv: list[str] | None = None) -> None:
    args = collect_args(argv)
    prepare_tau2(args.tau2_root)
    from tau2.registry import registry
    loader = registry.get_tasks_loader(args.task_set)
    # ``telecom_full`` is a zero-argument official full-pool loader, while
    # the regular ``telecom`` task set accepts task_split_name.
    split = None if args.task_split == "all" else args.task_split
    tasks = loader() if split is None else loader(task_split_name=split)
    # The official full pool is grouped by workflow/failure family.  Taking
    # its lexical prefix would yield a homogeneous S (e.g. mostly MMS
    # issues), so use a reproducible full-pool shuffle instead of selecting
    # easier or shorter tasks.
    tasks = [
        task for task in tasks
        if len((getattr(task, "evaluation_criteria", None).actions or [])) >= args.min_reference_actions
    ]
    tasks = sorted(tasks, key=lambda task: str(task.id))
    random.Random(args.selection_seed).shuffle(tasks)
    if args.target_count > len(tasks):
        raise RuntimeError(f"target {args.target_count} exceeds {len(tasks)} official tasks in {args.task_set}/{args.task_split}")
    root, traces = args.output_dir / "collection", args.output_dir / "collection" / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    accepted_path, rejected_path, progress_path = root / "accepted.jsonl", root / "rejected.jsonl", root / "progress.json"
    accepted = {str(row["task_id"]) for row in jsonl(accepted_path)} if args.resume else set()
    by_id = {str(task.id): task for task in tasks}
    retries = [str(row.get("task_id")) for row in jsonl(rejected_path)] if args.resume else []
    if args.resume and rejected_path.exists():
        rejected_path.unlink()
    pending, seen = [], set()
    for task_id in retries + [str(task.id) for task in tasks]:
        if task_id in by_id and task_id not in accepted and task_id not in seen:
            pending.append(by_id[task_id]); seen.add(task_id)
    active: dict[Any, Any] = {}; position = 0; failed = 0

    def progress(status: str) -> None:
        progress_path.write_text(json.dumps({"schema": "agentwm_tau2_collection_progress_v1", "status": status, "accepted": len(accepted), "target": args.target_count, "failed_this_run": failed, "pending": len(pending) - position, "active": len(active), "available": len(tasks), "domain": args.domain, "task_split": args.task_split, "selection": "full_pool_seeded_shuffle", "selection_seed": args.selection_seed, "min_reference_actions": args.min_reference_actions, "min_visible_tool_calls": args.min_tool_calls, "max_steps": args.max_steps, "model": args.proxy_model, "workers": args.workers}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def run(task: Any) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(args.task_retries + 1):
            try:
                return collect_one(args, task)
            except Exception as exc:
                last = exc
                if attempt < args.task_retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"failed after {args.task_retries + 1} attempts: {last}")

    progress("running")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(accepted) < args.target_count:
            while len(active) < args.workers and len(accepted) + len(active) < args.target_count and position < len(pending):
                task = pending[position]; position += 1; active[pool.submit(run, task)] = task
            progress("running")
            if not active:
                break
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                try:
                    trace = future.result()
                    if trace["quality"]["tool_calls"] < args.min_tool_calls:
                        # A text-only answer is a valid model outcome but it
                        # contains no executable action for this trace corpus.
                        # Match the SWE collector: skip it without recording a
                        # failure, and let the next candidate fill the slot.
                        progress("running")
                        continue
                    if not trace["quality"]["trainable"]:
                        raise RuntimeError(
                            "trace has unpaired real assistant tool calls: "
                            f"calls={trace['quality']['tool_calls']}, "
                            f"paired={trace['quality']['paired_tool_calls']}, "
                            f"unpaired={trace['quality']['unpaired_tool_calls']}"
                        )
                    path = traces / f"{task.id}.json"
                    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    accepted.add(str(task.id))
                    append_jsonl(accepted_path, {"task_id": str(task.id), "session_id": trace["session_id"], "termination_reason": trace["termination_reason"], "quality": trace["quality"], "trace": str(path)})
                except Exception as exc:
                    failed += 1; append_jsonl(rejected_path, {"task_id": str(task.id), "error": str(exc)})
                progress("running")
    progress("complete" if len(accepted) == args.target_count else "exhausted")
    if len(accepted) != args.target_count:
        raise RuntimeError(f"only collected {len(accepted)}/{args.target_count} Tau2 traces")


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] == "collect":
        collect(argv[1:] if argv else None); return
    if argv[0] == "freeze":
        _freeze_selection(argv[1:]); return
    if argv[0] == "robustness":
        from robustness import run_benchmark
        run_benchmark("tau2", argv[1:]); return
    raise SystemExit(f"unknown Tau2 command {argv[0]!r}; use collect, freeze, or robustness")


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


if __name__ == "__main__":
    main()
