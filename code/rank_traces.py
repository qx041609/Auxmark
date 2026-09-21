#!/usr/bin/env python3
"""Rank source traces from paired probe labels and detector replay results."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from output_layout import BENCHES, TEACHERS, DEFAULT_ROOT, area_dir, trace_dir


WEIGHTS = (0.25, 0.1, 0.1)
TOP_KS = (10, 15, 20, 25, 30)


def read_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}:{number}: missing {key}")
        if value in rows:
            raise ValueError(f"{path}:{number}: duplicate {key} {value!r}")
        rows[value] = row
    return rows


def trace_ids(path: Path) -> set[str]:
    """Read source/known trace IDs from a trace directory or JSONL file."""
    if path.is_dir():
        rows = [json.loads(file.read_text(encoding="utf-8")) for file in sorted(path.glob("*.json"))]
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {row.get("trace_id") for row in rows}
    if not ids or not all(isinstance(value, str) and value for value in ids) or len(ids) != len(rows):
        raise ValueError(f"{path}: trace_id must be present and unique in every trace")
    return ids


def source_sessions(path: Path, accepted_path: Path | None = None) -> tuple[set[str], dict[str, str]]:
    if path.is_dir():
        rows = [json.loads(file.read_text(encoding="utf-8")) for file in sorted(path.glob("*.json"))]
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if accepted_path:
        accepted = {
            (str(row.get("task_id") or ""), str(row.get("session_id") or ""))
            for row in (json.loads(line) for line in accepted_path.read_text(encoding="utf-8").splitlines() if line.strip())
        }
        if not accepted:
            raise ValueError(f"{accepted_path}: no accepted task/session pairs")
        rows = [row for row in rows if (str(row.get("task_id") or ""), str(row.get("session_id") or "")) in accepted]
        if len(rows) != len(accepted):
            raise ValueError(f"{path}: accepted task/session pairs do not match source traces")
    ids = {row.get("trace_id") for row in rows}
    if not ids or not all(isinstance(value, str) and value for value in ids) or len(ids) != len(rows):
        raise ValueError(f"{path}: trace_id must be present and unique in selected source traces")
    sessions: dict[str, str] = {}
    for row in rows:
        session = row.get("session_id")
        if not isinstance(session, str) or not session or session in sessions:
            raise ValueError(f"{path}: session_id must be present and unique in every source trace")
        sessions[session] = row["trace_id"]
    return ids, sessions


def load_side(labels_path: Path, results_path: Path) -> dict[str, dict[str, Any]]:
    labels = read_jsonl(labels_path, "probe_id")
    results = read_jsonl(results_path, "probe_id")
    if labels.keys() != results.keys():
        missing = sorted(labels.keys() - results.keys())[:3]
        extra = sorted(results.keys() - labels.keys())[:3]
        raise ValueError(f"incomplete replay {results_path}: missing={missing}, extra={extra}")
    for probe_id, label in labels.items():
        result = results[probe_id]
        for field in ("card_id",):
            if not isinstance(label.get(field), str) or not label[field]:
                raise ValueError(f"{labels_path}: {probe_id}: missing {field}")
        if result.get("card_id") != label["card_id"]:
            raise ValueError(f"{results_path}: {probe_id}: card_id mismatch")
        for field in ("tool_hit", "full_hit"):
            if not isinstance(result.get(field), bool):
                raise ValueError(f"{results_path}: {probe_id}: {field} must be boolean")
        if result["full_hit"] and not result["tool_hit"]:
            raise ValueError(f"{results_path}: {probe_id}: full_hit without tool_hit")
    return {probe_id: {"label": label, "result": results[probe_id]} for probe_id, label in labels.items()}


def midranks(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    """Average rank for ties, divided by N+1 within this model/cohort."""
    ordered = sorted(rows, key=lambda row: (row[field], row["trace_id"]))
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][field] == ordered[i][field]:
            j += 1
        rank = ((i + 1) + j) / 2 / (len(ordered) + 1)
        for row in ordered[i:j]:
            ranks[row["trace_id"]] = rank
        i = j
    return ranks


def rank_traces(
    real: dict[str, dict[str, Any]], fake: dict[str, dict[str, Any]],
    allowed: set[str] | None = None, sessions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if real.keys() != fake.keys():
        raise ValueError("real and fake sides must contain exactly the same probe_id values")
    counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"real": [0, 0, 0], "fake": [0, 0, 0]})
    for probe_id in real:
        r, f = real[probe_id], fake[probe_id]
        for field in ("source_trace_id", "card_id", "expected_tool", "expected_arguments"):
            if r["label"].get(field) != f["label"].get(field):
                raise ValueError(f"{probe_id}: paired labels disagree on {field}")
        label = r["label"]
        direct_id = label.get("source_trace_id")
        evidence = label.get("release_evidence")
        session = evidence.get("session_id") if isinstance(evidence, dict) else None
        mapped_id = sessions.get(session) if sessions and isinstance(session, str) else None
        if direct_id and mapped_id and direct_id != mapped_id:
            raise ValueError(f"{probe_id}: source_trace_id disagrees with source session")
        trace_id = mapped_id or direct_id
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError(f"{probe_id}: no source_trace_id; pass --source-traces to map release_evidence.session_id")
        if allowed is not None and trace_id not in allowed:
            continue
        for side, item in (("real", r), ("fake", f)):
            hit = item["result"]
            total = counts[trace_id][side]
            total[0] += int(hit["full_hit"])
            total[1] += int(hit["tool_hit"] and not hit["full_hit"])
            total[2] += 1
    if not counts:
        raise ValueError("no source traces have paired probes")
    rows = []
    for trace_id, pair in sorted(counts.items()):
        r, f = pair["real"], pair["fake"]
        rows.append({
            "trace_id": trace_id,
            "real_probes": r[2], "fake_probes": f[2],
            "real_full_rate": r[0] / r[2], "virtual_full_rate": f[0] / f[2],
            "real_tool_only_rate": r[1] / r[2], "virtual_tool_only_rate": f[1] / f[2],
        })
    fields = ("real_full_rate", "virtual_full_rate", "real_tool_only_rate", "virtual_tool_only_rate")
    ranks = [midranks(rows, field) for field in fields]
    for row in rows:
        trace_id = row["trace_id"]
        row["score"] = (ranks[0][trace_id] - WEIGHTS[0] * ranks[1][trace_id]
                        + WEIGHTS[1] * ranks[2][trace_id] - WEIGHTS[2] * ranks[3][trace_id])
    rows.sort(key=lambda row: (-round(row["score"], 12), row["virtual_full_rate"],
                               -row["real_full_rate"], row["trace_id"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank suspicious source traces from paired replay results")
    for name in ("real-labels", "fake-labels", "real-results", "fake-results"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--bench", choices=BENCHES,
                        help="read inputs from output/<bench>/{trace,probes,evaluation}/<teacher>")
    parser.add_argument("--teacher", choices=TEACHERS)
    parser.add_argument("--model", help="student model directory under evaluation/<teacher>/<condition>")
    parser.add_argument("--condition", default="D_c")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-traces", type=Path, help="optional accepted source trace directory or JSONL")
    parser.add_argument("--accepted", type=Path, help="optional accepted.jsonl task/session manifest for retry-heavy collections")
    parser.add_argument("--known-traces", type=Path, help="optional known training traces, only for Precision@K")
    parser.add_argument("--known-dc", action="store_true", help="use this bench/teacher's D_c as evaluation labels")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.bench:
        if not args.teacher or not args.model:
            parser.error("--bench requires --teacher and --model")
        probe = area_dir(args.bench, "probes", args.teacher, args.output_root) / "S"
        replay = area_dir(args.bench, "evaluation", args.teacher, args.output_root) / args.condition / args.model
        args.real_labels = args.real_labels or probe / "real_probes" / "labels.jsonl"
        args.fake_labels = args.fake_labels or probe / "fake_probes" / "labels.jsonl"
        args.real_results = args.real_results or replay / "real_probes" / "results.jsonl"
        args.fake_results = args.fake_results or replay / "fake_probes" / "results.jsonl"
        args.source_traces = args.source_traces or trace_dir(args.bench, args.teacher, "S", args.output_root) / "standard_traces"
        if args.known_dc:
            args.known_traces = args.known_traces or trace_dir(args.bench, args.teacher, "D_c", args.output_root) / "standard_traces"
        args.out_dir = args.out_dir or replay / "trace_ranking"
    elif args.known_dc:
        parser.error("--known-dc requires --bench")
    for name in ("real_labels", "fake_labels", "real_results", "fake_results", "out_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required without --bench")
    if args.accepted and not args.source_traces:
        parser.error("--accepted requires --source-traces")
    real = load_side(args.real_labels, args.real_results)
    fake = load_side(args.fake_labels, args.fake_results)
    allowed, sessions = source_sessions(args.source_traces, args.accepted) if args.source_traces else (None, None)
    ranked = rank_traces(real, fake, allowed, sessions)
    summary: dict[str, Any] = {
        "method": "within_model_full_tool_only_midrank",
        "weights": list(WEIGHTS),
        "ranked_traces": len(ranked),
        "paired_probes": sum(row["real_probes"] for row in ranked),
        "excluded_unscorable_source_traces": len(allowed - {row["trace_id"] for row in ranked}) if allowed else None,
    }
    if args.known_traces:
        known = trace_ids(args.known_traces)
        for row in ranked:
            row["known_training_trace"] = row["trace_id"] in known
        summary["precision_at_k"] = {
            str(k): sum(row["known_training_trace"] for row in ranked[:k]) / k
            for k in TOP_KS if k <= len(ranked)
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "trace_ranking.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ranked), encoding="utf-8"
    )
    (args.out_dir / "ranking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
