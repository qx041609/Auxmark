"""Canonical per-benchmark output paths shared by public entry points."""
from __future__ import annotations

from pathlib import Path


BENCHES = ("BFCL", "SWEbench", "Tau2")
TEACHERS = ("GPT", "Kimi")
AREAS = ("trace", "probes", "training", "evaluation")
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "output"


def area_dir(bench: str, area: str, teacher: str, root: Path = DEFAULT_ROOT) -> Path:
    if bench not in BENCHES or area not in AREAS or teacher not in TEACHERS:
        raise ValueError(f"invalid output selection: {bench}/{area}/{teacher}")
    return root / bench / area / teacher


def trace_dir(bench: str, teacher: str, condition: str, root: Path = DEFAULT_ROOT) -> Path:
    if not condition or "/" in condition or "\\" in condition or condition in {".", ".."}:
        raise ValueError(f"invalid condition: {condition!r}")
    return area_dir(bench, "trace", teacher, root) / condition


def collection_dir(bench: str, teacher: str, root: Path = DEFAULT_ROOT) -> Path:
    return area_dir(bench, "trace", teacher, root) / "_collection_work"
