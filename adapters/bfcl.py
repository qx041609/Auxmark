#!/usr/bin/env python3
"""BFCL adapter: collection, S/D_c freezing, and robustness-data construction.

This is deliberately a thin benchmark adapter: watermark scheduling and trace
recording stay in ``embedding.py``/``watermark_proxy.py``.  The adapter only loads
official BFCL tasks, executes returned calls in BFCL's in-memory sandbox, and
feeds observations back to the proxy.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import deque
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfcl-root", type=Path, required=True)
    parser.add_argument("--teacher", choices=("GPT", "Kimi"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--proxy-base-url", required=True)
    parser.add_argument("--proxy-model", default="agentwm-watermarked-teacher")
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--task-retries", type=int, default=4,
                        help="Retry a transport/model failure this many additional times before final rejection.")
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument(
        "--task-ids-file", type=Path,
        help="Optional newline-delimited task-id subset, for an auditable retry pass.",
    )
    parser.add_argument("--account-id", default="bfcl-agentwm")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.teacher:
        args.output_dir = args.output_dir or args.output_root / "BFCL" / "trace" / args.teacher / "_collection_work" / "collection"
    if args.output_dir is None:
        parser.error("--output-dir or --teacher is required")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalise_schema(value: Any) -> Any:
    """Make BFCL's Python-oriented JSON schemas valid OpenAI/Google schemas.

    BFCL function docs use ``dict`` in places where OpenAI expects ``object``.
    A few nested schemas also carry ``properties`` without an explicit type;
    Google rejects those declarations even though other providers tolerate
    them.  This is only schema syntax normalisation: names, fields, required
    sets, and descriptions are unchanged.
    """
    if isinstance(value, list):
        return [normalise_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: normalise_schema(item) for key, item in value.items()}
    # BFCL's Python docs use a few non-JSON-Schema type names.  ModelArk's
    # Kimi tool decoder validates schemas strictly and rejects ``float`` as
    # invalid decoding guidance; ``number`` is its standards-equivalent JSON
    # Schema type.  This preserves every parameter name and value domain.
    if result.get("type") == "dict":
        result["type"] = "object"
    elif result.get("type") == "float":
        result["type"] = "number"
    if isinstance(result.get("properties"), dict) and "type" not in result:
        result["type"] = "object"
    return result


def tools_for(root: Path, classes: list[str]) -> list[dict[str, Any]]:
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING

    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    doc_dir = root / "bfcl_eval" / "data" / "multi_turn_func_doc"
    for class_name in classes:
        for item in read_jsonl(doc_dir / MULTI_TURN_FUNC_DOC_FILE_MAPPING[class_name]):
            name = str(item["name"])
            if name in names:
                raise RuntimeError(f"ambiguous function name {name!r}")
            names.add(name)
            parameters = normalise_schema(item.get("parameters") or {"type": "object", "properties": {}})
            rows.append({"type": "function", "function": {
                "name": name, "description": str(item.get("description") or ""), "parameters": parameters,
            }})
    return rows


def request(base_url: str, payload: dict[str, Any], retries: int = 4, timeout: float = 360) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    # The proxy persists state by this header.  Forward the collector's
    # unique task session so concurrent/similar prompts cannot overwrite
    # another task's watermark trajectory or evidence.
    headers = {"Content-Type": "application/json"}
    if session_id := str(payload.get("session_id") or "").strip():
        headers["X-AgentWM-Session"] = session_id
    http = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                  headers=headers, method="POST")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(http, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:800]
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            # HTTP 4xx other than gateway throttling is a deterministic
            # request problem; retrying would merely duplicate the call.
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"request failed after {retries + 1} attempts: {last_error}") from last_error


def normalise_call(raw: dict[str, Any], index: int) -> dict[str, Any]:
    function = raw.get("function") or {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("tool-call arguments are not an object")
    return {"id": str(raw.get("id") or f"call_{index}"), "type": "function",
            "function": {"name": str(function.get("name") or ""), "arguments": json.dumps(arguments, ensure_ascii=False)}}


def python_call(call: dict[str, Any], allowed: set[str]) -> str:
    name = call["function"]["name"]
    if name not in allowed:
        raise ValueError(f"model selected a function absent from task tool table: {name!r}")
    arguments = json.loads(call["function"]["arguments"])
    return name + "(" + ",".join(f"{key}={value!r}" for key, value in arguments.items()) + ")"


def task_trace(root: Path, base_url: str, model: str, account: str, task: dict[str, Any], max_steps: int) -> dict[str, Any]:
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

    tools = tools_for(root, list(task["involved_classes"]))
    allowed = {tool["function"]["name"] for tool in tools}
    session_id = f"bfcl-{task['id']}-{int(time.time_ns())}"
    messages: list[dict[str, Any]] = []
    steps = 0
    for turn in task["question"]:
        messages.extend(turn)
        while steps < max_steps:
            payload = {
                "model": model, "messages": messages, "tools": tools, "tool_choice": "auto", "temperature": 0,
                "max_tokens": 768, "session_id": session_id, "user": account,
                "benchmark": "bfcl", "domain": "multi_turn", "task_id": task["id"],
            }
            response = request(base_url, payload)
            raw_message = response.get("choices", [{}])[0].get("message") or {}
            raw_calls = raw_message.get("tool_calls") or []
            assistant: dict[str, Any] = {"role": "assistant", "content": raw_message.get("content")}
            if not raw_calls:
                messages.append(assistant)
                break
            calls = [normalise_call(call, steps * 100 + index) for index, call in enumerate(raw_calls)]
            assistant["tool_calls"] = calls
            messages.append(assistant)
            executable = [python_call(call, allowed) for call in calls]
            results, _ = execute_multi_turn_func_call(
                executable, task["initial_config"], task["involved_classes"], session_id, task["id"],
                long_context="long_context" in task["id"] or "composite" in task["id"],
            )
            for call, result in zip(calls, results):
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})
            steps += 1
        if steps >= max_steps:
            # A bounded collection run has a valid trajectory prefix even when
            # the benchmark conversation would continue.  Keep that prefix,
            # including all already-recorded watermark events, rather than
            # discarding it as an exception.
            return {"schema": "agentwm_bfcl_collection_trace_v1", "task_id": task["id"], "session_id": session_id,
                    "messages": messages, "tool_schemas": tools, "steps": steps,
                    "termination_reason": "max_steps"}
    return {"schema": "agentwm_bfcl_collection_trace_v1", "task_id": task["id"], "session_id": session_id,
            "messages": messages, "tool_schemas": tools, "steps": steps,
            "termination_reason": "completed"}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def collect(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.target_count < 1 or args.workers < 1:
        raise ValueError("target count and workers must be positive")
    root = args.bfcl_root.resolve()
    sys.path.insert(0, str(root))
    data_dir = root / "bfcl_eval" / "data"
    tasks: list[dict[str, Any]] = []
    for category in [value.strip() for value in args.categories.split(",") if value.strip()]:
        tasks.extend(read_jsonl(data_dir / f"BFCL_v4_{category}.json"))
    tasks.sort(key=lambda row: str(row["id"]))
    if args.task_ids_file:
        wanted = {
            line.strip() for line in args.task_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        tasks = [task for task in tasks if str(task["id"]) in wanted]
        missing = wanted - {str(task["id"]) for task in tasks}
        if missing:
            raise RuntimeError(f"task ids not found in selected categories: {sorted(missing)[:3]}")
    if args.target_count > len(tasks):
        raise RuntimeError(f"target {args.target_count} exceeds {len(tasks)} available tasks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = args.output_dir / "traces"
    trace_dir.mkdir(exist_ok=True)
    accepted_path, progress_path = args.output_dir / "accepted.jsonl", args.output_dir / "progress.json"
    accepted = {json.loads(line)["task_id"] for line in accepted_path.read_text(encoding="utf-8").splitlines()} if args.resume and accepted_path.exists() else set()
    pending = [task for task in tasks if task["id"] not in accepted]

    def persist_progress(status: str) -> None:
        progress_path.write_text(json.dumps({"status": status, "accepted": len(accepted), "target": args.target_count,
                                             "available": len(tasks)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    persist_progress("running")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(pending)
        retry_queue: deque[dict[str, Any]] = deque()
        attempts: dict[str, int] = {}
        active: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
        while len(accepted) < args.target_count:
            # Reserve a target slot for every active request.  Without this,
            # several concurrent successes can land after the target is
            # reached and turn a 100-trace run into 103/107 traces.
            while len(active) < args.workers and len(accepted) + len(active) < args.target_count:
                try:
                    task = retry_queue.popleft() if retry_queue else next(iterator)
                except StopIteration:
                    break
                active[pool.submit(task_trace, root, args.proxy_base_url, args.proxy_model, args.account_id, task, args.max_steps)] = task
            if not active:
                break
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                try:
                    trace = future.result()
                    path = trace_dir / f"{task['id']}.json"
                    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    accepted.add(task["id"])
                    append_jsonl(accepted_path, {"task_id": task["id"], "session_id": trace["session_id"], "trace": str(path)})
                except Exception as exc:
                    task_id = str(task["id"])
                    attempts[task_id] = attempts.get(task_id, 0) + 1
                    if attempts[task_id] <= args.task_retries:
                        retry_queue.append(task)
                    else:
                        append_jsonl(args.output_dir / "rejected.jsonl", {
                            "task_id": task_id, "error": str(exc), "attempts": attempts[task_id],
                        })
                persist_progress("running")
                if len(accepted) >= args.target_count:
                    break
    persist_progress("complete" if len(accepted) >= args.target_count else "exhausted")
    if len(accepted) < args.target_count:
        raise RuntimeError(f"only collected {len(accepted)}/{args.target_count} traces")


def main() -> None:
    """Single public entrypoint used by BFCL shell launchers.

    ``collect`` owns retry-and-skip behaviour: a failed task is retried up to
    ``--task-retries`` times, then recorded as rejected and replaced by the
    next candidate.  There is deliberately no post-hoc retry/merge command.
    """
    argv = sys.argv[1:]
    if not argv or argv[0] == "collect":
        collect(argv[1:] if argv else None)
        return
    command, rest = argv[0], argv[1:]
    if command == "freeze":
        _freeze_selection(rest)
        return
    if command == "robustness":
        from robustness import run_benchmark
        run_benchmark("bfcl", rest)
        return
    raise SystemExit(f"unknown BFCL command {command!r}; use collect, freeze, or robustness")


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
