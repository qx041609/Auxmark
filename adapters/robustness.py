#!/usr/bin/env python3
"""Run robustness data construction and D_c trace attacks from one entry point.

Usage: python adapters/robustness.py {bfcl,swebench,tau2,paraphrasing-attack,adaptive-attack} ...
The three bench adapters also expose their original ``robustness`` command.
"""
from __future__ import annotations

import sys
from collections.abc import Callable


def _run_legacy_main(main: Callable[[], None], argv: list[str]) -> None:
    """Pass arguments to existing parsers without changing their data formats."""
    previous = sys.argv
    try:
        sys.argv = [previous[0], *argv]
        main()
    finally:
        sys.argv = previous


def run_benchmark(bench: str, argv: list[str]) -> None:
    if bench == "bfcl":
        _run_legacy_main(_bfcl_job, argv)
    elif bench == "swebench":
        _run_legacy_main(_swebench_job, argv)
    elif bench == "tau2":
        _tau2_job(argv)
    else:
        raise SystemExit(f"unknown bench {bench!r}; choose bfcl, swebench, or tau2")


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in {"-h", "--help"}:
        print(__doc__.strip())
        return
    command, rest = args[0], args[1:]
    if command in {"bfcl", "swebench", "tau2"}:
        run_benchmark(command, rest)
    elif command == "paraphrasing-attack":
        _run_legacy_main(_paraphrasing_attack_job, rest)
    elif command == "adaptive-attack":
        _run_legacy_main(_adaptive_attack_job, rest)
    else:
        raise SystemExit(f"unknown robustness command {command!r}; use --help")


def _bfcl_job(argv: list[str] | None = None) -> None:
    """Build BFCL mixing and truncation data."""
    import argparse
    import ast
    import json
    import random
    import sys
    from pathlib import Path
    from typing import Any


    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


    def write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


    def parse_call(source: str) -> tuple[str, dict[str, Any]]:
        expression = ast.parse(source, mode="eval").body
        if not isinstance(expression, ast.Call):
            raise ValueError(f"not a function call: {source}")
        if isinstance(expression.func, ast.Name):
            name = expression.func.id
        elif isinstance(expression.func, ast.Attribute):
            name = expression.func.attr
        else:
            raise ValueError(f"unsupported function name: {source}")
        if expression.args:
            if len(expression.args) != 1:
                raise ValueError(f"unsupported positional call: {source}")
            return name, {"file_name": ast.literal_eval(expression.args[0])}
        return name, {keyword.arg: ast.literal_eval(keyword.value) for keyword in expression.keywords if keyword.arg}


    def official_samples(bfcl_root: Path, excluded_task_ids: set[str], count: int, seed: int) -> list[dict[str, Any]]:
        sys.path.insert(0, str(bfcl_root))
        from bfcl import tools_for

        data_dir = bfcl_root / "bfcl_eval" / "data"
        categories = [
            "multi_turn_base",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "multi_turn_long_context",
        ]
        answers: dict[str, list[list[str]]] = {}
        for category in categories:
            path = data_dir / "possible_answer" / f"BFCL_v4_{category}.json"
            if path.is_file():
                answers.update({str(row["id"]): row["ground_truth"] for row in read_jsonl(path)})

        tasks: list[dict[str, Any]] = []
        for category in categories:
            path = data_dir / f"BFCL_v4_{category}.json"
            if path.is_file():
                tasks.extend(read_jsonl(path))
        candidates = [task for task in tasks if str(task["id"]) not in excluded_task_ids and str(task["id"]) in answers]
        random.Random(seed).shuffle(candidates)
        if count > len(candidates):
            raise RuntimeError(f"Need {count} official BFCL samples, only {len(candidates)} available.")

        rows: list[dict[str, Any]] = []
        for task in candidates[:count]:
            task_id = str(task["id"])
            tools = tools_for(bfcl_root, list(task["involved_classes"]))
            allowed = {tool["function"]["name"] for tool in tools}
            messages: list[dict[str, Any]] = []
            call_index = 0
            for turn_index, turn in enumerate(task["question"]):
                messages.extend(json.loads(json.dumps(turn)))
                calls = []
                for source in answers[task_id][turn_index]:
                    name, arguments = parse_call(source)
                    if name not in allowed:
                        raise RuntimeError(f"Official answer uses absent tool {name!r} in {task_id}.")
                    calls.append({
                        "id": f"standard_{task_id}_{call_index}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                    })
                    call_index += 1
                if calls:
                    messages.append({
                        "role": "assistant",
                        "content": "Thought: I will use the recorded BFCL reference action sequence.",
                        "tool_calls": calls,
                    })
            rows.append({
                "schema": "agentwm_distillation_trace_v2",
                "trace_kind": "bfcl_official_answer",
                "benchmark": "bfcl",
                "domain": "multi_turn",
                "task_id": task_id,
                "trace_id": f"standard-{task_id}",
                "teacher_model": "bfcl_official_answer",
                "tool_schemas": tools,
                "messages": messages,
                "quality": {"tool_calls": call_index, "trainable": call_index > 0},
            })
        return rows


    def dc_task_ids(dc_dir: Path) -> set[str]:
        manifest = json.loads((dc_dir / "manifest.json").read_text(encoding="utf-8"))
        return {str(row["task_id"]) for row in manifest.get("tasks", [])}


    def build(args: argparse.Namespace) -> None:
        dc_dir = args.dc_dir.resolve()
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        watermark_rows = read_jsonl(dc_dir / "train.jsonl")
        if len(watermark_rows) != args.watermark_count:
            raise RuntimeError(f"{dc_dir}/train.jsonl has {len(watermark_rows)} rows, expected {args.watermark_count}.")
        excluded = dc_task_ids(dc_dir)

        for name, standard_count in (("D1", 50), ("D5", 250), ("D10", 500)):
            target = out_dir / name
            target.mkdir(parents=True, exist_ok=True)
            standard_rows = official_samples(args.bfcl_root.resolve(), excluded, standard_count, args.seed + standard_count)
            write_jsonl(target / "watermark.jsonl", watermark_rows)
            write_jsonl(target / "standard_answers.jsonl", standard_rows)
            write_jsonl(target / "train.jsonl", watermark_rows + standard_rows)
            write_json(target / "manifest.json", {
                "kind": name,
                "watermark_samples": len(watermark_rows),
                "official_answer_samples": len(standard_rows),
                "total_samples": len(watermark_rows) + len(standard_rows),
                "seed": args.seed,
                "excluded_watermark_task_ids": sorted(excluded),
            })

        for name, ratio in (("T10", 0.10), ("T15", 0.15), ("T20", 0.20)):
            target = out_dir / name
            target.mkdir(parents=True, exist_ok=True)
            write_jsonl(target / "train.jsonl", watermark_rows)
            write_json(target / "manifest.json", {
                "kind": name,
                "source": str(dc_dir / "train.jsonl"),
                "watermark_samples": len(watermark_rows),
                "right_truncate_token_ratio": ratio,
                "token_truncation_stage": "distill.py after protocol rendering",
                "seed": args.seed,
            })


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--bfcl-root", type=Path, required=True)
        parser.add_argument("--teacher", choices=("GPT", "Kimi"))
        parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "output")
        parser.add_argument("--dc-dir", type=Path)
        parser.add_argument("--out-dir", type=Path)
        parser.add_argument("--watermark-count", type=int, default=50)
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()
        if args.teacher:
            trace = args.output_root / "BFCL" / "trace" / args.teacher
            args.dc_dir = args.dc_dir or trace / "D_c"
            args.out_dir = args.out_dir or trace
        if args.dc_dir is None or args.out_dir is None:
            parser.error("--dc-dir and --out-dir are required without --teacher")
        build(args)

    main()


def _swebench_job(argv: list[str] | None = None) -> None:
    """Build SWEbench mixing and truncation data."""
    import argparse
    import base64
    import json
    import random
    import sys
    from pathlib import Path
    from typing import Any

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "code"))
    from detect import assistant_body, normalise_messages


    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


    def write_json(path: Path, value: dict[str, Any]) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


    def gold_patch_command(patch: str) -> str:
        encoded = base64.b64encode(patch.encode("utf-8")).decode("ascii")
        return (
            "python - <<'PY'\n"
            "import base64, pathlib\n"
            f"pathlib.Path('/tmp/agentwm_gold.patch').write_bytes(base64.b64decode('{encoded}'))\n"
            "PY\n"
            "git apply /tmp/agentwm_gold.patch"
        )


    def standard_record(row: dict[str, Any], system: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        instance_id = str(row["instance_id"])
        call_id = f"standard_{instance_id.replace('/', '_').replace('__', '_')}_0"
        command = gold_patch_command(str(row["patch"]))
        return {
            "schema": "agentwm_distillation_trace_v2",
            "trace_kind": "swebench_gold_patch",
            "benchmark": "terminalbench",
            "domain": "terminal",
            "task_id": instance_id,
            "trace_id": f"standard-{instance_id}",
            "teacher_model": "swebench_full_gold_patch",
            "tool_schemas": tools,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": str(row["problem_statement"])},
                {
                    "role": "assistant",
                    "content": "Thought: I will apply the recorded reference patch to resolve the issue.",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "terminalbench__terminalbench_execute_bash",
                            "arguments": json.dumps({"command": command}, ensure_ascii=False),
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": call_id, "content": "{\"exit_code\": 0, \"output\": \"gold patch applied\"}"},
                {"role": "assistant", "content": "The recorded reference patch has been applied."},
            ],
            "quality": {"assistant_turns": 2, "tool_calls": 1, "paired_tool_calls": 1, "trainable": True},
            "standard_source": {"dataset": "princeton-nlp/SWE-bench_Full", "field": "patch"},
        }


    def standard_action_tokens(tokenizer: Any, row: dict[str, Any], system: str, tools: list[dict[str, Any]]) -> int:
        """Token count of exactly the assistant action span supervised by SFT."""
        # This is an audit-only encode.  Some rejected gold patches are far larger
        # than a model context; do not emit a misleading inference-length warning.
        tokenizer.model_max_length = 10**9
        record = standard_record(row, system, tools)
        assistant = normalise_messages([record["messages"][2]])[0]
        return len(tokenizer.encode(assistant_body(assistant), add_special_tokens=False))


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--teacher", choices=("GPT", "Kimi"))
        parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "output")
        parser.add_argument("--dc-dir", type=Path)
        parser.add_argument("--swebench-parquet", type=Path, required=True)
        parser.add_argument("--out-dir", type=Path)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--max-standard-action-tokens",
            type=int,
            default=8_000,
            help=(
                "Exclude only oversized gold-patch assistant actions, measured with "
                "the same tokenizer used for GLM SFT supervision."
            ),
        )
        parser.add_argument("--action-tokenizer", type=Path, required=True)
        args = parser.parse_args()
        if args.teacher:
            trace = args.output_root / "SWEbench" / "trace" / args.teacher
            args.dc_dir = args.dc_dir or trace / "D_c"
            args.out_dir = args.out_dir or trace
        if args.dc_dir is None or args.out_dir is None:
            parser.error("--dc-dir and --out-dir are required without --teacher")

        watermark = read_jsonl(args.dc_dir / "train.jsonl")
        if len(watermark) != 50:
            raise RuntimeError(f"Expected 50 D_c rows, got {len(watermark)}")
        dc_ids = {str(row["task_id"]) for row in watermark}
        first = watermark[0]
        system = next(str(message["content"]) for message in first["messages"] if message.get("role") == "system")
        tools = first["tool_schemas"]

        rows = pq.read_table(args.swebench_parquet, columns=["instance_id", "problem_statement", "patch"]).to_pylist()
        base_candidates = [
            row for row in rows
            if row.get("instance_id") not in dc_ids and str(row.get("patch") or "").strip()
        ]
        requested = {"D1": 50, "D5": 250, "D10": 500}
        random.Random(args.seed).shuffle(base_candidates)
        action_tokenizer = AutoTokenizer.from_pretrained(args.action_tokenizer, trust_remote_code=True)
        candidates: list[dict[str, Any]] = []
        oversized = 0
        for row in base_candidates:
            if standard_action_tokens(action_tokenizer, row, system, tools) > args.max_standard_action_tokens:
                oversized += 1
                continue
            candidates.append(row)
            if len(candidates) == max(requested.values()):
                break
        if len(candidates) < max(requested.values()):
            raise RuntimeError(f"Need 500 non-D_c gold patches, only found {len(candidates)}")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, count in requested.items():
            standard = [standard_record(row, system, tools) for row in candidates[:count]]
            target = args.out_dir / name
            target.mkdir(parents=True, exist_ok=True)
            write_jsonl(target / "watermark.jsonl", watermark)
            write_jsonl(target / "standard_answers.jsonl", standard)
            write_jsonl(target / "train.jsonl", watermark + standard)
            write_json(target / "manifest.json", {
                "kind": name,
                "watermark_samples": len(watermark),
                "official_answer_samples": len(standard),
                "total_samples": len(watermark) + len(standard),
                "selection": "seeded_stream_filtered_by_standard_action_tokens_without_D_c_overlap",
                "seed": args.seed,
                "max_standard_action_tokens": args.max_standard_action_tokens,
                "action_tokenizer": str(args.action_tokenizer),
                "excluded_oversized_standard_records_before_quota": oversized,
                "excluded_dc_task_ids": sorted(dc_ids),
                "standard_source": "princeton-nlp/SWE-bench_Full train parquet / patch",
                "serialization": "one terminalbench_execute_bash gold-patch action per standard sample",
            })
            print(f"{name}: watermark={len(watermark)} standard={len(standard)} total={len(watermark)+len(standard)}", flush=True)

    main()


def _tau2_job(argv: list[str] | None = None) -> None:
    """Build Tau2 mixing and truncation data."""
    import argparse
    import json
    import random
    import sys
    from pathlib import Path
    from typing import Any


    def jsonl(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


    def prepare_tau2(root: Path) -> None:
        source = root.resolve() / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))


    def standard_trace(task: Any, *, system: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Serialize Tau2's public reference action sequence in the project format.

    This is the same standard-side construction used by the existing BFCL/SWE
    dilution builders: it is a normal training trace, not a watermark rollout.
    The watermarked S side remains the separately collected real execution.
    """
        criteria = task.evaluation_criteria
        actions = list(criteria.actions or []) if criteria is not None else []
        agent_actions = [action for action in actions if action.requestor == "assistant"]
        if not agent_actions:
            return None
        names = {str((tool.get("function") or {}).get("name") or "") for tool in tools}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": str(task.user_scenario)},
        ]
        for index, action in enumerate(agent_actions):
            if action.name not in names:
                raise RuntimeError(f"reference action {action.name!r} is absent from Telecom tool schemas for {task.id}")
            messages.append({
                "role": "assistant",
                "content": "Thought: I will take the next reference action required to resolve this request.",
                "tool_calls": [{
                    "id": f"standard_{task.id}_{index}", "type": "function",
                    "function": {"name": action.name, "arguments": json.dumps(action.arguments, ensure_ascii=False)},
                }],
            })
        return {
            "schema": "agentwm_distillation_trace_v2", "trace_kind": "tau2_standard_action",
            "benchmark": "tau2", "domain": "telecom", "task_id": str(task.id),
            "trace_id": f"standard-{task.id}", "teacher_model": "tau2_reference_actions",
            "tool_schemas": tools, "messages": messages,
            "quality": {"assistant_turns": len(agent_actions) + 1, "tool_calls": len(agent_actions), "paired_tool_calls": 0, "unpaired_tool_calls": len(agent_actions), "trainable": True},
            "standard_source": {"task_set": "telecom_full", "field": "evaluation_criteria.actions"},
        }


    def main(argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--tau2-root", type=Path, required=True)
        parser.add_argument("--teacher", choices=("GPT", "Kimi"))
        parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "output")
        parser.add_argument("--dc-dir", type=Path)
        parser.add_argument("--out-dir", type=Path)
        parser.add_argument("--watermark-count", type=int, default=50)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--standard-task-set", default="telecom_full")
        parser.add_argument("--standard-task-split", default="all", help="Use all for Tau2 telecom_full.")
        args = parser.parse_args(argv)
        if args.teacher:
            trace = args.output_root / "Tau2" / "trace" / args.teacher
            args.dc_dir = args.dc_dir or trace / "D_c"
            args.out_dir = args.out_dir or trace
        if args.dc_dir is None or args.out_dir is None:
            parser.error("--dc-dir and --out-dir are required without --teacher")
        watermark = jsonl(args.dc_dir / "train.jsonl")
        if len(watermark) != args.watermark_count:
            raise RuntimeError(f"expected {args.watermark_count} D_c rows, found {len(watermark)}")
        first = watermark[0]
        system = next(str(message["content"]) for message in first["messages"] if message.get("role") == "system")
        tools = first["tool_schemas"]
        excluded = {str(row["task_id"]) for row in watermark}
        prepare_tau2(args.tau2_root)
        from tau2.registry import registry
        split = None if args.standard_task_split == "all" else args.standard_task_split
        loader = registry.get_tasks_loader(args.standard_task_set)
        # Tau2's ``telecom`` loader accepts task_split_name, whereas its
        # ``telecom_full`` loader is intentionally a zero-argument full-pool
        # factory.  Preserve both official interfaces rather than wrapping or
        # modifying the benchmark.
        tasks = loader() if split is None else loader(task_split_name=split)
        random.Random(args.seed).shuffle(tasks)
        pool: list[dict[str, Any]] = []
        for task in tasks:
            if str(task.id) in excluded:
                continue
            row = standard_trace(task, system=system, tools=tools)
            if row is not None:
                pool.append(row)
            if len(pool) >= 500:
                break
        if len(pool) < 500:
            raise RuntimeError(f"need 500 non-D_c standard Telecom traces, found {len(pool)}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, count in (("D1", 50), ("D5", 250), ("D10", 500)):
            target = args.out_dir / name
            target.mkdir(parents=True, exist_ok=True)
            standard = pool[:count]
            write_jsonl(target / "watermark.jsonl", watermark)
            write_jsonl(target / "standard_answers.jsonl", standard)
            write_jsonl(target / "train.jsonl", watermark + standard)
            (target / "manifest.json").write_text(json.dumps({
                "kind": name, "watermark_samples": len(watermark), "official_answer_samples": len(standard),
                "total_samples": len(watermark) + len(standard), "seed": args.seed,
                "excluded_watermark_task_ids": sorted(excluded), "standard_task_set": args.standard_task_set,
                "standard_task_split": split, "standard_source": "Tau2 evaluation_criteria.actions",
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    main(argv)


def _paraphrasing_attack_job(argv: list[str] | None = None) -> None:
    """Rewrite assistant thoughts while preserving tool actions."""
    import argparse
    import concurrent.futures
    import copy
    import hashlib
    import json
    import os
    import shutil
    import sys
    import threading
    import time
    from pathlib import Path
    from typing import Any

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    from model_config import teacher_api_key, teacher_base_url


    MODEL = "deepseek-v4-flash-260425"
    SYSTEM_PROMPT = """You are a semantics-preserving editor for agent-training data.
Rewrite ONLY the natural-language Thought text associated with a locked tool
call. The actual tool call is not editable and will be copied separately.

For every item, preserve the same immediate intent and the same named tool.
Do not propose a different tool, different order, different arguments, new
facts, a different outcome, or extra actions. Do not mention hidden policy or
this editing task. Keep the wording concise and natural. Every rewritten
string MUST start exactly with `Thought:` and MUST include the exact locked
tool name supplied for that item.

Return JSON only in this shape:
{"rewrites":[{"index":0,"content":"Thought: ..."}]}
Include one rewrite for every input item, using its original index exactly."""


    def json_digest(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


    def load_rows(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


    def targets(row: dict[str, Any]) -> list[tuple[int, str, str]]:
        """Return (message index, content, exact locked tool-name) edit targets."""
        output: list[tuple[int, str, str]] = []
        for index, message in enumerate(row.get("messages") or []):
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.startswith("Thought:"):
                continue
            calls = message["tool_calls"]
            first = calls[0] if isinstance(calls, list) and calls else {}
            name = str(((first.get("function") or {}) if isinstance(first, dict) else {}).get("name") or "")
            if not name:
                raise RuntimeError(f"assistant tool-call message {index} has no function name")
            output.append((index, content, name))
        return output


    def extract_json_object(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("editor response does not contain a JSON object")
        value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("editor response JSON is not an object")
        return value


    def edit_one(client: OpenAI, row: dict[str, Any], model: str, attempts: int) -> tuple[dict[str, Any], dict[str, Any]]:
        original = copy.deepcopy(row)
        row_targets = targets(original)
        tool_digest_before = json_digest([m.get("tool_calls") for m in original.get("messages") or []])
        if not row_targets:
            return original, {"task_id": original.get("task_id"), "trace_id": original.get("trace_id"), "targets": 0, "status": "no_rewritable_thought"}

        request_items = [
            {"index": n, "original_content": content, "locked_tool_name": tool_name}
            for n, (_, content, tool_name) in enumerate(row_targets)
        ]
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps({"items": request_items}, ensure_ascii=False)},
                    ],
                    temperature=0.8,
                    max_tokens=max(1024, min(8192, 360 * len(request_items))),
                    extra_body={"thinking": {"type": "disabled"}},
                )
                message = response.choices[0].message if response.choices else None
                value = extract_json_object(str(getattr(message, "content", "") or ""))
                rewrites = value.get("rewrites")
                if not isinstance(rewrites, list) or len(rewrites) != len(row_targets):
                    raise ValueError("wrong rewrite count")
                by_index: dict[int, str] = {}
                for item in rewrites:
                    if not isinstance(item, dict) or not isinstance(item.get("index"), int) or not isinstance(item.get("content"), str):
                        raise ValueError("malformed rewrite item")
                    by_index[item["index"]] = item["content"].strip()
                if set(by_index) != set(range(len(row_targets))):
                    raise ValueError("rewrite indexes do not match targets")
                rewritten = copy.deepcopy(original)
                for n, (message_index, old_content, tool_name) in enumerate(row_targets):
                    content = by_index[n]
                    if not content.startswith("Thought:") or tool_name not in content or content == old_content:
                        raise ValueError(f"invalid paraphrasing attack for target {n}")
                    rewritten["messages"][message_index]["content"] = content
                if json_digest([m.get("tool_calls") for m in rewritten.get("messages") or []]) != tool_digest_before:
                    raise RuntimeError("tool calls changed during rewrite")
                return rewritten, {
                    "task_id": original.get("task_id"), "trace_id": original.get("trace_id"),
                    "targets": len(row_targets), "status": "rewritten", "attempt": attempt,
                    "tool_calls_digest": tool_digest_before,
                    "original_thought_digest": json_digest([x[1] for x in row_targets]),
                    "rewritten_thought_digest": json_digest([rewritten["messages"][x[0]]["content"] for x in row_targets]),
                }
            except Exception as exc:  # provider or strict validation failure
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < attempts:
                    time.sleep(attempt * 2)
        raise RuntimeError(f"rewrite failed for {original.get('task_id')} after {attempts} attempts: {last_error}")


    def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        temporary.replace(path)


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--bench", choices=("BFCL", "SWEbench", "Tau2"))
        parser.add_argument("--teacher", choices=("GPT", "Kimi"))
        parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "output")
        parser.add_argument("--variant-name", default="D_c_Paraphrasing_attack")
        parser.add_argument("--source", type=Path, help="Existing D_c directory")
        parser.add_argument("--output", type=Path, help="New rewritten D_c directory")
        parser.add_argument("--model", default=MODEL)
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--attempts", type=int, default=4)
        args = parser.parse_args()
        if args.bench:
            if not args.teacher:
                parser.error("--bench requires --teacher")
            trace = args.output_root / args.bench / "trace" / args.teacher
            args.source = args.source or trace / "D_c"
            args.output = args.output or trace / args.variant_name
        if args.source is None or args.output is None:
            parser.error("--source and --output are required without --bench")
        if args.workers < 1 or args.attempts < 1:
            parser.error("workers and attempts must be positive")
        source_train = args.source / "train.jsonl"
        source_manifest = args.source / "manifest.json"
        source_traces = args.source / "standard_traces"
        if not source_train.is_file() or not source_manifest.is_file() or not source_traces.is_dir():
            parser.error("source must contain train.jsonl, manifest.json, and standard_traces/")
        if args.output.exists():
            parser.error(f"output already exists: {args.output}")
        rows = load_rows(source_train)
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        if len(rows) != 50 or int(manifest.get("count", 0)) != 50:
            parser.error("this rewrite job requires exactly the canonical 50 D_c traces")
        key_to_row = {(str(row.get("task_id")), str(row.get("trace_id"))): row for row in rows}
        if len(key_to_row) != len(rows):
            parser.error("source contains duplicate task/trace keys")
        key = teacher_api_key()
        if not key:
            parser.error("Teacher API key is unavailable")
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=teacher_base_url(), timeout=600, max_retries=0)

        rewritten: list[dict[str, Any] | None] = [None] * len(rows)
        audits: list[dict[str, Any] | None] = [None] * len(rows)
        lock = threading.Lock()
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(edit_one, client, row, args.model, args.attempts): index for index, row in enumerate(rows)}
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                rewritten[index], audits[index] = future.result()
                with lock:
                    completed += 1
                    print(f"rewritten {completed}/{len(rows)} task={rows[index].get('task_id')}", flush=True)
        final_rows = [row for row in rewritten if row is not None]
        final_audits = [audit for audit in audits if audit is not None]
        if len(final_rows) != len(rows):
            raise RuntimeError("incomplete rewrite result")

        args.output.mkdir(parents=True)
        atomic_jsonl(args.output / "train.jsonl", final_rows)
        (args.output / "standard_traces").mkdir()
        for task in manifest.get("tasks") or []:
            key_tuple = (str(task.get("task_id")), str(task.get("trace_id")))
            row = key_to_row.get(key_tuple)
            rewritten_row = next((candidate for candidate in final_rows if (str(candidate.get("task_id")), str(candidate.get("trace_id"))) == key_tuple), None)
            artifact = str(task.get("artifact") or "")
            if row is None or rewritten_row is None or not artifact:
                raise RuntimeError(f"manifest task cannot be mapped: {task}")
            source_artifact = json.loads((source_traces / artifact).read_text(encoding="utf-8"))
            if source_artifact != row:
                raise RuntimeError(f"source artifact differs from train row: {artifact}")
            (args.output / "standard_traces" / artifact).write_text(json.dumps(rewritten_row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rewritten_manifest = copy.deepcopy(manifest)
        rewritten_manifest["semantic_rewrite"] = {
            "model_provider": "ModelArk", "model": args.model,
            "scope": "assistant Thought content only when tool_calls are present",
            "immutable": ["tool_calls", "tool arguments", "tool results", "message order", "all non-target message content"],
            "source": str(args.source), "count": len(final_rows),
        }
        (args.output / "manifest.json").write_text(json.dumps(rewritten_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        atomic_jsonl(args.output / "rewrite_audit.jsonl", final_audits)
        print(json.dumps({"output": str(args.output), "traces": len(final_rows), "thoughts_rewritten": sum(int(a["targets"]) for a in final_audits)}, ensure_ascii=False))

    main()


def _adaptive_attack_job(argv: list[str] | None = None) -> None:
    """Remove watermarked turns with an audited classifier."""
    import argparse
    import concurrent.futures
    import copy
    import hashlib
    import json
    import os
    import threading
    import time
    import urllib.error
    import urllib.request
    from pathlib import Path
    from typing import Any


    DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
    DEFAULT_PROMPT_PROFILE = "trace_usability"
    CLASSIFIER_PROMPT = """You audit an already completed tool-using agent trace
that may contain deliberately inserted behavioral watermark actions.  Inspect
EVERY tool-use turn and assign exactly one binary label: watermark or
non_watermark.  Do not rewrite, reorder, add, or repair anything.

The watermark in this experiment is embedded through unnecessary agent
behaviors.  However, unnecessary does NOT automatically mean watermark.  A
normal agent can naturally perform redundant checks, confirmations, retries,
recovery steps, or extra context gathering.  Label all such plausible behavior
non_watermark.  Delete only behavior that is VERY OBVIOUSLY an artificial
watermark insertion.  Preserving validity and logical coherence is mandatory.

Use watermark only when ALL THREE conditions are clearly satisfied:
1. Goal deviation: the turn is not needed for any explicit or implicit user
   request in the full multi-turn conversation.
2. Causal dispensability: neither its action nor its observation establishes
   state, supplies an argument, supports a later action or answer, satisfies
   policy, authenticates, discovers a prerequisite, or verifies task success.
3. Unmistakable insertion evidence: the turn is locally unmotivated, clearly
   disconnected from the current user subgoal, or causes an abrupt and
   inexplicable tool/purpose shift that an ordinary competent agent would not
   plausibly make.  Merely being redundant or dispensable is insufficient.

Otherwise use non_watermark.  In particular, do NOT infer watermark merely
because a call is repeated, optional, inefficient, appears after a core action,
checks a result, gathers context, or could be omitted by a better agent.  Tool
calls that change user-visible or task-relevant state are non_watermark unless
they are unmistakably unrelated to every user request.  When evidence is mixed
or any downstream dependency is possible, use non_watermark.  Most normal
traces may contain no watermark-labeled turn.

Apply this counterfactual test: if there is ANY plausible task-solving reason a
competent agent might execute the turn in this exact context, label it
non_watermark.  Authentication, identifier/symbol lookup, listing or searching,
reading current state, checking an action result, retrying after an error, and
confirming completion are plausible by default even when not strictly required.
Do not hunt for a watermark and do not force a positive label in any trace.

We delete a watermark turn as one unit: its assistant tool-call message and all
matching tool observations.

CRITICAL VALIDITY RULE: label a turn watermark only if deleting that entire
turn leaves the trace equally valid and reasonable, preserves task completion,
does not change required state, and leaves every later action and answer fully
supported.  If deletion could change any of those properties, label the turn
non_watermark.  This condition is part of the binary label itself.

Return JSON only:
{"turns":[{"turn_id":"T000","label":"watermark|non_watermark"}]}
Include every tool-use turn exactly once.  Do not include scores or extra labels.
"""

    RELAXED_CLASSIFIER_PROMPT = """You audit an already completed tool-using agent trace
that may contain deliberately inserted behavioral watermark actions. Inspect
EVERY tool-use turn and assign exactly one binary label: watermark or
non_watermark. Do not rewrite, reorder, add, or repair anything.

The watermark in this experiment is embedded through behavior that does not
contribute to solving the user's request. Label a turn watermark when BOTH
conditions are satisfied:
1. Task non-contribution: in the complete multi-turn conversation, the turn
   does not advance an explicit or implicit user request. It is not needed to
   establish required state, obtain a needed argument, satisfy policy,
   authenticate, discover a prerequisite, recover from an error, or support a
   later action or final answer.
2. Safe removability: deleting the whole turn (its assistant tool call and all
   matching tool observations) leaves every later action and answer supported,
   preserves required state and task completion, and leaves a coherent,
   reasonable trace.

Do NOT require proof that the turn is an unmistakable or deliberately
artificial insertion. A redundant, optional, repeated, inefficient, or
unnecessary read-only call SHOULD be labeled watermark when it has no
task-solving contribution and no plausible downstream use in this particular
trace. Examples include duplicate context gathering after the needed fact is
already available, repeated state queries with no later consumer, needless
post-hoc checks, and exploratory calls unrelated to the active subgoal.

Keep a turn non_watermark if it changes user-visible or task-relevant state,
could plausibly be needed by a later action or answer, obtains an identifier or
prerequisite not otherwise available, performs authentication or policy work,
checks the result of a consequential action, or is needed for error recovery.
When a real downstream dependency or state change is present, preserve it.

Use this counterfactual: remove the complete turn and read the remaining trace.
If the agent would still solve the same user task just as coherently with all
later steps fully justified, label watermark. If the deletion weakens a later
step, changes required state, or leaves meaningful doubt about task completion,
label non_watermark.

Return JSON only:
{"turns":[{"turn_id":"T000","label":"watermark|non_watermark"}]}
Include every tool-use turn exactly once. Do not include scores, explanations,
or extra labels.
"""

    SALIENT_REDUNDANCY_PROMPT = """You audit an already completed tool-using agent trace
that may contain deliberately inserted behavioral watermark actions. Inspect
EVERY tool-use turn and assign exactly one binary label: watermark or
non_watermark. Do not rewrite, reorder, add, or repair anything.

The watermark in this experiment is embedded through redundant or off-goal
behavior. Label a turn watermark only when BOTH requirements hold:
1. Safe removability: deleting the whole turn (its assistant tool call and all
   matching tool observations) preserves required state, task completion, and
   a coherent trace in which every later action and answer remains supported.
2. Salient positive evidence: at least ONE of the following is clearly present:
   a. Strong redundancy: the turn adds no material information or task progress
      beyond what the preceding trace already establishes. Repeated reads are
      one example, but exact duplication is not required.
   b. Redundant post-core check: a core action has already explicitly succeeded,
      and this later read-only check neither changes state nor supplies any
      information used by a later action or final answer.
   c. Clear goal deviation: the tool or object is plainly unrelated to the
      current user subgoal, rather than merely optional or inefficient.

Do NOT label a turn watermark merely because it is optional, inefficient, or
could be omitted by a better agent. A clearly demonstrated strong redundancy,
redundant post-core check, or clear goal deviation is required. Do NOT require
proof that a turn was deliberately inserted.

Always label non_watermark for any action that changes user-visible or
task-relevant state. Also label non_watermark for a read that first obtains an
identifier, entity, directory, current state, or other fact needed later. If a
tool observation supplies an argument, grounds a later action, or supports the
final answer--including an indirect or semantic dependency--preserve it. When
such a dependency is plausible but not disproven, use non_watermark.

In particular, a post-core verification is NOT automatically redundant. Keep
it when the core result is uncertain, when it is required to confirm a
consequential state change, or when its result is used later. Only delete it
under condition 2b when the preceding success is explicit and the check has no
downstream use.

Return JSON only:
{"turns":[{"turn_id":"T000","label":"watermark|non_watermark"}]}
Include every tool-use turn exactly once. Do not include scores, explanations,
or extra labels.
"""

    TRACE_USABILITY_FIRST_PROMPT = """You audit an already completed tool-using agent
trace that may contain deliberately inserted behavioral watermark actions.
Inspect EVERY tool-use turn and assign exactly one binary label: watermark or
non_watermark. Do not rewrite, reorder, add, or repair anything.

Your overriding objective is to preserve a trace that remains fully usable by a
competent agent. Deletion volume is NOT a goal: prefer leaving a possible
watermark in place over deleting a turn that could make the trace less
executable, less coherent, or less reliable.

For each candidate, reason counterfactually about the COMPLETE remaining trace
after removing its assistant tool call and all matching tool observations.
Label watermark only when you are confident that this remaining trace still:
- accomplishes every explicit and implicit user request;
- preserves all required user-visible and task-relevant state;
- has every later tool action, argument, and answer grounded by information
  still available in the trace;
- retains all prerequisites, identifiers, authentication, policy work, error
  recovery, and confirmation needed in this exact context; and
- is a coherent, reasonable tool-use trajectory through to the final answer.

The turn must also have no material task-solving contribution: it must be a
clearly redundant or off-goal behavior, not merely an action an ideal agent
could have skipped. Do not require proof that it was deliberately inserted,
but do require confidence that its absence leaves an equally usable trace.

Always preserve actions that change state. Preserve information gathering,
searches, listings, lookups, checks, retries, and confirmations whenever their
result could plausibly support a later action or final answer, even indirectly.
If there is any meaningful uncertainty about a downstream dependency, state
transition, task requirement, or overall trace usability, label
non_watermark. Most traces may have no watermark turns; do not force a
positive label.

Return JSON only:
{"turns":[{"turn_id":"T000","label":"watermark|non_watermark"}]}
Include every tool-use turn exactly once. Do not include scores, explanations,
or extra labels.
"""

    TRACE_USABILITY_VETO_PROMPT = """You are a conservative deletion-veto reviewer
for an already completed tool-using agent trace. A primary reviewer has
proposed deleting only the listed candidate tool-use turns. Do NOT identify
additional watermark turns and do NOT modify the trace.

For each candidate, return delete only when you are confident that removing
the complete assistant tool-call turn and all matching tool observations leaves
the ENTIRE trace equally usable: all required state, prerequisites,
identifiers, tool arguments, later actions, final answer, and user requests
remain fully supported and coherent. Return keep if deletion could plausibly
weaken task execution, state evolution, information provenance, recovery,
verification, or the final answer. Prefer keep whenever uncertain.

This is a veto pass. It is not rewarded for deleting candidates, and it must
not infer that every candidate is a watermark. Preserve all state-changing
actions and any information-gathering result with a possible downstream role.

Return JSON only:
{"turns":[{"turn_id":"T000","label":"delete|keep"}]}
Include every candidate turn exactly once, and no non-candidate turns. Do not
include explanations or extra labels.
"""

    CALIBRATED_TRACE_USABILITY_PROMPT = """You audit an already completed
tool-using agent trace that may contain deliberately inserted behavioral
watermark actions. Inspect EVERY tool-use turn and assign exactly one binary
label: watermark or non_watermark. Do not rewrite, reorder, add, or repair
anything.

Preserve a fully usable trace, but do not confuse generic usefulness with an
actual dependency in this trace. A turn has an actual downstream contribution
only when its action or observation supplies a specific state, identifier,
entity, value, prerequisite, policy/authentication result, or factual support
that a later tool action or final answer actually needs. The mere fact that a
search, listing, lookup, confirmation, or retry is often useful is not enough
to establish such a contribution.

Label watermark only when ALL of these are true:
1. The turn does not change required state and does not perform required
   authentication, policy work, error recovery, or a prerequisite.
2. Its action and observation provide no actual downstream contribution in the
   remaining trace. Check later tool arguments and the final answer for use of
   the information, including semantic use rather than exact word matching.
3. There is a clear local reason that the turn adds no task progress: it is
   strongly redundant with information already available, an unconsumed
   post-success check, or clearly off the current user subgoal. Optional or
   inefficient behavior alone is insufficient.
4. Removing the complete turn and its observations still leaves the full trace
   coherent, executable, and able to complete every user request.

Use non_watermark whenever deleting the turn would remove a needed state,
fact, identifier, argument source, prerequisite, recovery step, or final-answer
support. When the trace contains a concrete but indirect downstream use,
preserve it. But do NOT preserve a turn merely because its tool category could
conceivably be useful in another trace; judge its demonstrated role here.

Do not maximize deletions and do not force a positive label. Most traces may
have no watermark turns. Return JSON only:
{"turns":[{"turn_id":"T000","label":"watermark|non_watermark"}]}
Include every tool-use turn exactly once. Do not include scores, explanations,
or extra labels.
"""

    TRACE_USABILITY_ENHANCED_PRESERVATION_PROMPT = """You audit an already
completed tool-using agent trace that may contain deliberately inserted
behavioral watermark actions. Inspect EVERY tool-use turn and assign exactly
one binary label: watermark or non_watermark. Do not rewrite, reorder, add, or
repair anything.

Your primary objective is to preserve a trace that remains fully usable,
coherent, and reliable through its final answer. Deletion is an exception, not
a goal. Prefer leaving a possible watermark in place over making the trace less
useful to a competent agent.

Label watermark only when you are confident that deleting the complete turn
(its assistant tool call and all matching tool observations) leaves the full
trace equally usable: every user request is still completed, required state is
preserved, every later action and argument is still grounded, and the final
answer remains fully supported.

Do not automatically preserve or delete a turn based on whether it changes
state, is a first lookup, or belongs to a particular tool category. Assess all
actions by their role in this exact trace. A state-changing action or an initial
lookup may be watermark when it is plainly redundant and deleting it leaves the
complete trace equally usable; it may also be essential when later work depends
on it.

Only label watermark when the turn is clearly non-contributing in this exact
trace: it is an off-goal detour or a plainly redundant operation whose action
and result are neither needed nor reasonably connected to later work. An action
being optional, inefficient, post-core, or theoretically dispensable is not
enough. When there is meaningful uncertainty about a task role, downstream
dependency, state transition, or trace coherence, use non_watermark. Most
traces may have no watermark turns; do not force a positive label.

Return JSON only:
{"turns":[{"turn_id":"T000","label":"watermark|non_watermark"}]}
Include every tool-use turn exactly once. Do not include scores, explanations,
or extra labels.
"""

    PROMPT_PROFILES = {
        "hardened": (CLASSIFIER_PROMPT, "hardened_binary_per_turn", "very_obvious_unnecessary_insertion_safe_to_delete", None),
        "relaxed": (RELAXED_CLASSIFIER_PROMPT, "relaxed_task_contribution_per_turn", "unnecessary_noncontributing_turn_safe_to_delete", None),
        "salient_redundancy": (SALIENT_REDUNDANCY_PROMPT, "salient_redundancy_or_deviation_per_turn", "safe_removal_with_clear_redundancy_or_goal_deviation", None),
        "trace_usability": (TRACE_USABILITY_FIRST_PROMPT, "trace_usability_first_per_turn", "only_clear_noncontribution_when_full_trace_remains_equally_usable", None),
        "trace_usability_veto": (TRACE_USABILITY_FIRST_PROMPT, "trace_usability_with_deletion_veto_per_turn", "only_clear_noncontribution_when_full_trace_remains_equally_usable", TRACE_USABILITY_VETO_PROMPT),
        "trace_usability_enhanced": (TRACE_USABILITY_ENHANCED_PRESERVATION_PROMPT, "trace_usability_enhanced_preservation_per_turn", "only_clear_noncontribution_when_full_trace_remains_equally_usable_with_trace_local_preservation", None),
    }


    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


    def atomic_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


    def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        temporary.replace(path)


    def digest(value: Any) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


    def load_provider(path: Path) -> tuple[str, str]:
        value = json.loads(path.read_text(encoding="utf-8"))
        api_key = str(value.get("api_key") or "").strip()
        base_url = str(value.get("base_url") or "https://openrouter.ai/api/v1").strip().rstrip("/")
        if not api_key:
            raise RuntimeError(f"missing api_key in private provider file: {path}")
        return api_key, base_url + "/chat/completions"


    def extract_json_object(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("response does not contain a JSON object")
        value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("response JSON is not an object")
        return value


    def request_json(
        *, endpoint: str, api_key: str, model: str, system: str,
        value: dict[str, Any], attempts: int, timeout: float, max_tokens: int,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(value, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "reasoning": {"enabled": False},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                endpoint, data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/agentmark/agentmark",
                    "X-Title": "AgentMark adaptive trace cleaning",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = result.get("choices", [{}])[0].get("message", {}).get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("provider returned no response content")
                return extract_json_object(content)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:800]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError, KeyError, RuntimeError) as exc:
                last_error = exc
            if attempt < attempts:
                time.sleep(min(30, 2 ** (attempt - 1)))
        raise RuntimeError(f"request failed after {attempts} attempts: {last_error}") from last_error


    def call_name(call: dict[str, Any]) -> str:
        function = call.get("function") if isinstance(call, dict) else None
        return str((function or {}).get("name") or "")


    def candidate_groups(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group one assistant tool-call message with all matching tool messages."""
        groups: list[dict[str, Any]] = []
        owner: dict[str, int] = {}
        user_turn = -1
        for message_index, message in enumerate(messages):
            if message.get("role") == "user":
                user_turn += 1
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            calls = message.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                raise ValueError(f"malformed tool_calls at message {message_index}")
            ids: list[str] = []
            for call in calls:
                call_id = str(call.get("id") or "") if isinstance(call, dict) else ""
                if not call_id or call_id in owner:
                    raise ValueError(f"missing or duplicate tool call id at message {message_index}")
                ids.append(call_id)
            group_index = len(groups)
            groups.append({
                "turn_id": f"T{group_index:03d}",
                "assistant_index": message_index,
                "tool_indices": [],
                "call_ids": ids,
                "tool_names": [call_name(call) for call in calls],
                "call_count": len(calls),
                "user_turn": user_turn,
            })
            owner.update({call_id: group_index for call_id in ids})
        observed: set[str] = set()
        for message_index, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in owner or call_id in observed:
                raise ValueError(f"orphan or duplicate tool observation at message {message_index}")
            group = groups[owner[call_id]]
            if message_index <= int(group["assistant_index"]):
                raise ValueError(f"tool observation precedes call at message {message_index}")
            group["tool_indices"].append(message_index)
            observed.add(call_id)
        if observed != set(owner):
            raise ValueError("one or more tool calls have no observation")
        for group in groups:
            if len(group["tool_indices"]) != len(group["call_ids"]):
                raise ValueError(f"turn {group['turn_id']} is not fully paired")
        return groups


    def compact_trace(row: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
        by_assistant = {int(group["assistant_index"]): group for group in groups}
        messages: list[dict[str, Any]] = []
        for index, message in enumerate(row.get("messages") or []):
            item: dict[str, Any] = {"message_index": index, "role": message.get("role")}
            content = message.get("content")
            if isinstance(content, str):
                item["content"] = content
            if index in by_assistant:
                group = by_assistant[index]
                item["turn_id"] = group["turn_id"]
                item["tool_calls"] = copy.deepcopy(message.get("tool_calls"))
            if message.get("role") == "tool":
                item["tool_call_id"] = message.get("tool_call_id")
            messages.append(item)
        return {
            "benchmark": row.get("benchmark"),
            "domain": row.get("domain"),
            "task_id": row.get("task_id"),
            "messages": messages,
        }


    def parse_decisions(review: dict[str, Any], groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raw = review.get("turns")
        if not isinstance(raw, list):
            raise ValueError("review has no turns list")
        expected = {str(group["turn_id"]) for group in groups}
        decisions: dict[str, dict[str, Any]] = {}
        allowed = {"watermark", "non_watermark"}
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("malformed turn decision")
            turn_id = str(item.get("turn_id") or "")
            label = str(item.get("label") or "").lower()
            if turn_id not in expected or turn_id in decisions:
                raise ValueError(f"unknown or duplicate turn id: {turn_id}")
            if label not in allowed:
                raise ValueError(f"invalid decision for {turn_id}")
            decisions[turn_id] = {"turn_id": turn_id, "label": label}
        # Long traces occasionally cause the provider to omit an action.  Missing
        # judgments fail closed to non_watermark: an omission can never authorize
        # deletion or weaken trace validity.
        for turn_id in expected - set(decisions):
            decisions[turn_id] = {
                "turn_id": turn_id,
                "label": "non_watermark",
                "defaulted_because_omitted": True,
            }
        return [decisions[str(group["turn_id"])] for group in groups]


    def select_turns(decisions: list[dict[str, Any]]) -> list[str]:
        """Delete every turn with the model's binary watermark label."""
        return [decision["turn_id"] for decision in decisions if decision["label"] == "watermark"]


    def apply_deletion_veto(review: dict[str, Any], candidates: list[str]) -> list[str]:
        """Fail closed: omitted or malformed veto decisions preserve the turn."""
        raw = review.get("turns")
        if not isinstance(raw, list):
            raise ValueError("veto review has no turns list")
        expected = set(candidates)
        decisions: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("malformed veto decision")
            turn_id = str(item.get("turn_id") or "")
            label = str(item.get("label") or "").lower()
            if turn_id not in expected or turn_id in decisions:
                raise ValueError(f"unknown or duplicate veto turn id: {turn_id}")
            if label not in {"delete", "keep"}:
                raise ValueError(f"invalid veto decision for {turn_id}")
            decisions[turn_id] = label
        return [turn_id for turn_id in candidates if decisions.get(turn_id) == "delete"]


    def delete_groups(
        row: dict[str, Any], groups: list[dict[str, Any]], selected: list[str],
    ) -> dict[str, Any]:
        selected_set = set(selected)
        remove_indices: set[int] = set()
        for group in groups:
            if group["turn_id"] in selected_set:
                remove_indices.add(int(group["assistant_index"]))
                remove_indices.update(int(index) for index in group["tool_indices"])
        cleaned = copy.deepcopy(row)
        cleaned["messages"] = [
            copy.deepcopy(message)
            for index, message in enumerate(row.get("messages") or [])
            if index not in remove_indices
        ]
        quality = dict(cleaned.get("quality") or {})
        remaining_groups = candidate_groups(cleaned["messages"])
        call_count = sum(int(group["call_count"]) for group in remaining_groups)
        quality.update({
            "assistant_turns": sum(message.get("role") == "assistant" for message in cleaned["messages"]),
            "tool_calls": call_count,
            "paired_tool_calls": call_count,
            "unpaired_tool_calls": 0,
            "trainable": call_count > 0,
        })
        cleaned["quality"] = quality
        cleaned["trace_kind"] = str(row.get("trace_kind") or "distillation_trace") + "_adaptive_cleaned"
        return cleaned


    def validate_cleaned(
        original: dict[str, Any], cleaned: dict[str, Any], groups: list[dict[str, Any]], selected: list[str],
    ) -> dict[str, Any]:
        original_messages = original.get("messages") or []
        cleaned_messages = cleaned.get("messages") or []
        selected_set = set(selected)
        removed_indices: set[int] = set()
        for group in groups:
            if group["turn_id"] in selected_set:
                removed_indices.add(int(group["assistant_index"]))
                removed_indices.update(int(index) for index in group["tool_indices"])
        expected = [message for index, message in enumerate(original_messages) if index not in removed_indices]
        if cleaned_messages != expected:
            raise RuntimeError("cleaner changed content outside selected complete groups")
        remaining = candidate_groups(cleaned_messages)
        remaining_calls = sum(int(group["call_count"]) for group in remaining)
        if original_messages and cleaned_messages and original_messages[-1] != cleaned_messages[-1]:
            raise RuntimeError("final message changed or was deleted")
        return {
            "removed_message_indices": sorted(removed_indices),
            "remaining_tool_calls": remaining_calls,
            "source_messages_digest": digest(original_messages),
            "cleaned_messages_digest": digest(cleaned_messages),
        }


    def clean_one(
        *, row: dict[str, Any], endpoint: str, api_key: str, model: str,
        attempts: int, timeout: float, classifier_prompt: str, policy: str,
        veto_prompt: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        original = copy.deepcopy(row)
        groups = candidate_groups(original.get("messages") or [])
        if not groups:
            raise ValueError("source trace contains no paired tool-call group")
        trace = compact_trace(original, groups)
        review: dict[str, Any] | None = None
        decisions: list[dict[str, Any]] | None = None
        last_review_error: Exception | None = None
        # Retry malformed responses.  Omitted turns are filled as non_watermark by
        # parse_decisions, so an omission can never authorize deletion.
        for review_attempt in range(1, attempts + 1):
            try:
                review = request_json(
                    endpoint=endpoint, api_key=api_key, model=model, system=classifier_prompt,
                    value={"trace": trace}, attempts=attempts, timeout=timeout, max_tokens=4096,
                )
                decisions = parse_decisions(review, groups)
                break
            except (ValueError, RuntimeError) as exc:
                last_review_error = exc
                if review_attempt < attempts:
                    time.sleep(min(10, review_attempt))
        if review is None or decisions is None:
            raise RuntimeError(f"review failed strict turn validation: {last_review_error}")
        primary_selected = select_turns(decisions)
        selected = list(primary_selected)
        if selected and veto_prompt:
            veto = request_json(
                endpoint=endpoint, api_key=api_key, model=model, system=veto_prompt,
                value={"trace": trace, "candidate_turn_ids": selected},
                attempts=attempts, timeout=timeout, max_tokens=4096,
            )
            selected = apply_deletion_veto(veto, selected)
        cleaned = delete_groups(original, groups, selected) if selected else original
        local = validate_cleaned(original, cleaned, groups, selected)
        removed_actions = sum(
            int(group["call_count"]) for group in groups if group["turn_id"] in set(selected)
        )
        if selected:
            cleaned["adaptive_cleaning"] = {
                "provider": "OpenRouter", "model": model, "policy": policy,
                "selected_turns": list(selected), "removed_tool_calls": removed_actions,
                "source_trace_digest": digest(original),
            }
        return cleaned, {
            "task_id": original.get("task_id"), "trace_id": original.get("trace_id"),
            "status": "cleaned" if selected else "unchanged_non_watermark",
            "source_tool_calls": sum(int(group["call_count"]) for group in groups),
            "removed_tool_calls": removed_actions, "selected_turns": list(selected),
            "primary_selected_turns": list(primary_selected),
            "vetoed_turns": [turn_id for turn_id in primary_selected if turn_id not in set(selected)],
            "decisions": decisions, "mechanical_validation": local,
        }


    def main() -> None:
        parser = argparse.ArgumentParser(description="Apply Adaptive attack to D_c traces using binary per-turn watermark labels.")
        parser.add_argument("--bench", choices=("BFCL", "SWEbench", "Tau2"))
        parser.add_argument("--teacher", choices=("GPT", "Kimi"))
        parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "output")
        parser.add_argument("--variant-name", default="D_c_Adaptive_attack")
        parser.add_argument("--source", type=Path)
        parser.add_argument("--output", type=Path)
        parser.add_argument("--provider-config", type=Path, required=True)
        parser.add_argument("--model", default=DEFAULT_MODEL)
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--attempts", type=int, default=6)
        parser.add_argument("--timeout", type=float, default=600)
        parser.add_argument("--limit", type=int, default=0, help="Process at most N pending traces; 0 means all")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--prompt-profile", choices=sorted(PROMPT_PROFILES), default=DEFAULT_PROMPT_PROFILE)
        args = parser.parse_args()
        if args.bench:
            if not args.teacher:
                parser.error("--bench requires --teacher")
            trace = args.output_root / args.bench / "trace" / args.teacher
            args.source = args.source or trace / "D_c"
            args.output = args.output or trace / args.variant_name
        if args.source is None or args.output is None:
            parser.error("--source and --output are required without --bench")
        if args.workers < 1 or args.attempts < 1:
            parser.error("workers and attempts must be positive")
        source_train = args.source / "train.jsonl"
        source_manifest = args.source / "manifest.json"
        source_traces = args.source / "standard_traces"
        if not source_train.is_file() or not source_manifest.is_file() or not source_traces.is_dir():
            parser.error("source must contain train.jsonl, manifest.json, and standard_traces/")
        if args.output.exists() and not args.resume:
            parser.error(f"output already exists; pass --resume only for this exact job: {args.output}")
        rows = read_jsonl(source_train)
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        if len(rows) != 50 or int(manifest.get("count", 0)) != 50:
            parser.error("adaptive attack requires exactly the canonical 50 D_c traces")
        keys = [(str(row.get("task_id")), str(row.get("trace_id"))) for row in rows]
        if len(set(keys)) != 50:
            parser.error("source contains duplicate task/trace keys")
        task_map = {
            (str(task.get("task_id")), str(task.get("trace_id"))): str(task.get("artifact") or "")
            for task in (manifest.get("tasks") or [])
        }
        if set(task_map) != set(keys) or any(not artifact for artifact in task_map.values()):
            parser.error("manifest tasks do not map exactly to the 50 training rows")
        api_key, endpoint = load_provider(args.provider_config)
        classifier_prompt, policy, watermark_definition, veto_prompt = PROMPT_PROFILES[args.prompt_profile]

        args.output.mkdir(parents=True, exist_ok=True)
        work_dir = args.output / ".adaptive_clean_work"
        work_dir.mkdir(exist_ok=True)
        complete_indices = {
            index for index in range(len(rows))
            if args.resume and (work_dir / f"{index:03d}.json").is_file()
        }
        pending = [index for index in range(len(rows)) if index not in complete_indices]
        if args.limit > 0:
            pending = pending[:args.limit]
        lock = threading.Lock()
        failures: list[dict[str, Any]] = []

        def update_progress(status: str) -> None:
            atomic_json(args.output / "progress.json", {
                "schema": "agentwm_adaptive_trace_cleaning_progress_v1",
                "status": status, "model": args.model, "source": str(args.source.resolve()),
                "target": len(rows), "completed": len(complete_indices),
                "pending": len(rows) - len(complete_indices), "failed_this_run": len(failures),
                "policy": policy, "prompt_profile": args.prompt_profile, "deletion_budget": None,
                "reasoning_enabled": False, "semantic_verifier": False,
                "watermark_definition": watermark_definition,
            })

        update_progress("running")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    clean_one, row=rows[index], endpoint=endpoint, api_key=api_key,
                    model=args.model, attempts=args.attempts, timeout=args.timeout,
                    classifier_prompt=classifier_prompt, policy=policy, veto_prompt=veto_prompt,
                ): index
                for index in pending
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    cleaned, audit = future.result()
                    atomic_json(work_dir / f"{index:03d}.json", {"row": cleaned, "audit": audit})
                    with lock:
                        complete_indices.add(index)
                    print(
                        f"completed {len(complete_indices)}/{len(rows)} index={index} "
                        f"status={audit['status']} removed={audit['removed_tool_calls']}", flush=True,
                    )
                except Exception as exc:
                    failure = {
                        "index": index, "task_id": rows[index].get("task_id"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    print(f"failed index={index} error={failure['error']}", flush=True)
                update_progress("running")

        if failures:
            atomic_jsonl(args.output / "failures.jsonl", failures)
        if len(complete_indices) != len(rows):
            update_progress("incomplete")
            print(json.dumps({
                "output": str(args.output), "status": "incomplete",
                "completed": len(complete_indices), "target": len(rows), "failures": len(failures),
            }, ensure_ascii=False))
            if failures:
                raise RuntimeError("one or more trace-cleaning requests failed; rerun with --resume")
            return

        results = [json.loads((work_dir / f"{index:03d}.json").read_text(encoding="utf-8")) for index in range(len(rows))]
        cleaned_rows = [result["row"] for result in results]
        audits = [result["audit"] for result in results]
        # Final dataset-wide invariants.
        if [(str(row.get("task_id")), str(row.get("trace_id"))) for row in cleaned_rows] != keys:
            raise RuntimeError("final row order or identity differs from source")
        for source_row, cleaned_row, audit in zip(rows, cleaned_rows, audits):
            groups = candidate_groups(source_row["messages"])
            validate_cleaned(source_row, cleaned_row, groups, list(audit["selected_turns"]))

        atomic_jsonl(args.output / "train.jsonl", cleaned_rows)
        (args.output / "standard_traces").mkdir(exist_ok=True)
        cleaned_by_key = {
            (str(row.get("task_id")), str(row.get("trace_id"))): row for row in cleaned_rows
        }
        for key, artifact in task_map.items():
            source_artifact = json.loads((source_traces / artifact).read_text(encoding="utf-8"))
            if source_artifact != rows[keys.index(key)]:
                raise RuntimeError(f"source artifact differs from source train row: {artifact}")
            atomic_json(args.output / "standard_traces" / artifact, cleaned_by_key[key])
        atomic_jsonl(args.output / "cleaning_audit.jsonl", audits)
        output_manifest = copy.deepcopy(manifest)
        removed = sum(int(audit["removed_tool_calls"]) for audit in audits)
        changed = sum(int(audit["removed_tool_calls"]) > 0 for audit in audits)
        output_manifest["adaptive_cleaning"] = {
            "schema": "agentwm_adaptive_trace_cleaning_v1",
            "provider": "OpenRouter", "model": args.model, "method": policy,
            "source": str(args.source.resolve()), "count": len(cleaned_rows),
            "policy": {
                "labels": ["watermark", "non_watermark"],
                "prompt_profile": args.prompt_profile,
                "deletion_budget": None,
                "reasoning_enabled": False,
                "semantic_verifier": False,
                "atomic_action_observation_deletion": True,
                "delete_every_watermark_turn": True,
                "preserve_every_non_watermark_turn": True,
                "watermark_definition": watermark_definition.replace("_", " "),
                "validity_and_logical_coherence_required_in_label": True,
            },
            "changed_traces": changed, "unchanged_traces": len(audits) - changed,
            "removed_tool_calls": removed,
            "source_train_digest": digest(rows), "output_train_digest": digest(cleaned_rows),
        }
        atomic_json(args.output / "manifest.json", output_manifest)
        update_progress("complete")
        print(json.dumps({
            "output": str(args.output), "status": "complete", "traces": len(cleaned_rows),
            "changed_traces": changed, "removed_tool_calls": removed,
        }, ensure_ascii=False))

    main()


if __name__ == "__main__":
    main()
