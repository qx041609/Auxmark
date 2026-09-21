#!/usr/bin/env python3
"""LoRA SFT for AgentWM trajectories with assistant-only loss.

Each source trajectory is one SFT sample. The native chat template renders the
whole multi-turn context (including tool observations), while labels are masked
outside assistant spans.  By default every assistant turn is supervised:
ordinary natural-language replies, Thought and tool actions.  This preserves
multi-turn interaction policies such as asking a user for a missing argument.
User/system prompts and tool observations are always context-only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from detect import PROTOCOL_NAME, flatten_for_local, render_protocol
from output_layout import BENCHES, TEACHERS, DEFAULT_ROOT, area_dir, trace_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-family", choices=("auto", "glm", "mistral"), default="auto",
                        help="Retained run metadata for GLM/Mistral launch compatibility; tool serialization is model-independent.")
    parser.add_argument("--bench", choices=BENCHES)
    parser.add_argument("--teacher", choices=TEACHERS)
    parser.add_argument("--condition", default="D_c", help="D_c reads D_c/train.jsonl; other names read trace/<condition>/train.jsonl")
    parser.add_argument("--run-name", help="student directory name under training/<teacher>/<condition>")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument(
        "--right-truncate-token-ratio",
        type=float,
        default=0.0,
        help="Delete this fraction from the right side of each rendered token sequence before max-length windowing.",
    )
    parser.add_argument("--epochs", type=float, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    # Keep the previous unified launcher defaults.  This file is only its
    # single-script replacement; it must not silently change the experiment.
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--moe",
        "--glm-moe-expert-lora",
        dest="glm_moe_expert_lora",
        action="store_true",
        help=(
            "Enable the existing GLM MoE expert Parameter-LoRA branch: also attach LoRA to the routed "
            "expert tensors `experts.gate_up_proj` and `experts.down_proj`. "
            "This requires --lora-dropout 0. `--glm-moe-expert-lora` remains an alias."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument(
        "--assistant-loss-mode",
        choices=("all", "thought_action"),
        default="all",
        help=(
            "Which assistant turns receive loss. `all` (default) supervises every "
            "assistant reply, Thought and tool call; `thought_action` is the legacy "
            "sparse policy that excludes ordinary assistant replies."
        ),
    )
    parser.add_argument(
        "--sparse-loss",
        action="store_true",
        help="Project the vocabulary head only at supervised next-token positions."
             " This avoids the [sequence,vocab] logits peak for long GLM runs.",
    )
    parser.add_argument(
        "--sparse-loss-ce-chunk-size",
        type=int,
        default=0,
        help=(
            "Compute sparse cross-entropy in this many supervised positions at "
            "a time (0 keeps one fused call).  This preserves the loss while "
            "avoiding the long-span softmax workspace peak."
        ),
    )
    parser.add_argument(
        "--max-supervised-tokens",
        type=int,
        default=0,
        help=(
            "Cap assistant-loss tokens within each trace after context windowing "
            "(0 disables the cap).  When capped, retain the most recent supervised "
            "Thought/Action tokens as labels and keep all preceding text as context."
        ),
    )
    parser.add_argument(
        "--mlp-sequence-chunk-size",
        type=int,
        default=0,
        help="Split position-independent MLPs over the sequence dimension during"
             " checkpoint recomputation (0 disables it).",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
        help="Explicit attention backend; sdpa lets PyTorch dispatch its Flash kernel when supported.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dynamic-batch-under-length",
        type=int,
        default=0,
        help="Use two examples per batch when their post-truncation lengths are below this threshold; longer examples stay singleton.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Resume Trainer state from a checkpoint directory instead of restarting from step zero.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help=(
            "Keep checkpoint-N directories after a successful final adapter save. "
            "By default completed runs delete them; interrupted or failed runs always retain them for resume."
        ),
    )
    parser.add_argument(
        "--deepspeed-config",
        default="",
        help="Optional DeepSpeed JSON configuration for sharded multi-GPU training.",
    )
    args = parser.parse_args()
    if args.bench:
        if not args.teacher or not args.run_name:
            parser.error("--bench requires --teacher and --run-name")
        args.data = args.data or str(trace_dir(args.bench, args.teacher, args.condition, args.output_root) / "train.jsonl")
        args.output_dir = args.output_dir or str(area_dir(args.bench, "training", args.teacher, args.output_root) / args.condition / args.run_name)
    if not args.data or not args.output_dir:
        parser.error("--data and --output-dir are required without --bench")
    return args


def all_linear_target_suffixes(model: torch.nn.Module) -> list[str]:
    """PEFT 0.19 lacks the newer ``all-linear`` shortcut.

    Enumerating suffixes is equivalent for these decoder-only models while
    deliberately excluding the output head, matching PEFT's all-linear
    convention and keeping the adapter focused on agent behaviour.
    """
    parameter_suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_parameters()}
    candidates = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] != "lm_head"
    }
    # GLM's routed-expert down projections are raw parameters named
    # ``down_proj``.  PEFT 0.19 would then match both those parameters and the
    # ordinary shared-expert Linear modules, which is unsafe with dropout.
    return sorted(candidates - parameter_suffixes)


def glm_moe_parameter_suffixes(model: torch.nn.Module) -> list[str]:
    """Return all non-attention GLM MLP/router weights for Parameter-LoRA.

    GLM stores each routed expert bank as a 3-D ``nn.Parameter`` rather than
    an ``nn.Linear``.  The ordinary all-linear discovery therefore cannot
    adapt it.  PEFT's ``target_parameters`` path is specifically intended for
    this MoE representation and accepts suffix matching, so the two suffixes
    All GLM MLP paths use this mechanism, including normal ``nn.Linear`` MLP
    weights.  Keeping MLPs out of ``target_modules`` avoids PEFT collapsing a
    suffix such as ``down_proj`` onto the routed-expert parameter and silently
    omitting the shared/dense MLP adapters.
    """
    parameter_names = {name for name, _ in model.named_parameters()}
    expected = {
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "shared_experts.gate_proj.weight",
        "shared_experts.up_proj.weight",
        "shared_experts.down_proj.weight",
        "experts.gate_up_proj",
        "experts.down_proj",
        "mlp.gate.weight",
    }
    missing = {suffix for suffix in expected if not any(name.endswith(suffix) for name in parameter_names)}
    if missing:
        raise RuntimeError(f"GLM MoE expert parameters not found: {sorted(missing)}")
    return sorted(expected)


def protocol_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=True)
    ids = encoded["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    index = 0
    while index < min(len(left), len(right)) and left[index] == right[index]:
        index += 1
    return index


def token_span_for_char_span(tokenizer: Any, full_ids: list[int], text: str, start: int, end: int) -> tuple[int, int]:
    """Map a rendered assistant payload to tokens without fast-tokenizer offsets.

    Mistral's official tokenizer intentionally exposes no offset mapping.  The
    longest stable prefix of encodings before/through the payload gives the
    same boundary and safely includes a boundary-merged token when necessary.
    """
    token_start = longest_common_prefix(full_ids, protocol_ids(tokenizer, text[:start]))
    token_end = longest_common_prefix(full_ids, protocol_ids(tokenizer, text[:end]))
    if token_end <= token_start:
        raise RuntimeError(f"Assistant payload has no token span: chars {start}:{end}")
    return token_start, token_end


def build_examples(
    tokenizer: Any,
    data_path: str,
    max_length: int,
    assistant_loss_mode: str = "all",
    right_truncate_token_ratio: float = 0.0,
    max_supervised_tokens: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "trajectories": 0,
        "assistant_turns": 0,
        "thought_action_turns": 0,
        "supervised_assistant_turns": 0,
        "kept_trace_examples": 0,
        "skipped_no_supervised_tokens": 0,
        "truncated_trace_examples": 0,
        "right_truncated_trace_examples": 0,
        "right_truncated_tokens_removed": 0,
        "supervision_capped_trace_examples": 0,
        "supervision_tokens_masked_by_cap": 0,
    }
    examples: list[dict[str, Any]] = []

    with open(data_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            stats["trajectories"] += 1
            messages = record["messages"]
            tools = record.get("tool_schemas")
            if not isinstance(tools, list) or not tools:
                raise RuntimeError(f"Trace {record.get('trace_id', line_number)!r} has no tool_schemas.")
            rendered = render_protocol(messages, tools, generation=False)
            prompt, character_spans = flatten_for_local(rendered)
            full_ids = protocol_ids(tokenizer, prompt)
            labels = [-100] * len(full_ids)
            if record.get("supervise_last_assistant_only"):
                # Decision-point SFT: the preceding history is context only;
                # exactly the final assistant Thought/Action is the target.
                character_spans = character_spans[-1:]
            for char_start, char_end, actionable in character_spans:
                if char_end <= char_start:
                    continue
                stats["assistant_turns"] += 1
                if actionable:
                    stats["thought_action_turns"] += 1
                if assistant_loss_mode != "all" and not actionable:
                    # Plain assistant replies are context-only under the
                    # legacy thought_action policy.  Do not try to derive a
                    # token span for them: a provider may legitimately emit a
                    # punctuation-only acknowledgement whose character span
                    # is absorbed into a neighbouring tokenizer token.
                    continue
                start, end = token_span_for_char_span(
                    tokenizer, full_ids, prompt, char_start, char_end
                )
                stats["supervised_assistant_turns"] += 1
                labels[start:end] = full_ids[start:end]

            if right_truncate_token_ratio:
                keep = max(1, int(len(full_ids) * (1.0 - right_truncate_token_ratio)))
                if keep < len(full_ids):
                    stats["right_truncated_trace_examples"] += 1
                    stats["right_truncated_tokens_removed"] += len(full_ids) - keep
                    full_ids = full_ids[:keep]
                    labels = labels[:keep]

            window_start = max(0, len(full_ids) - max_length)
            if window_start:
                stats["truncated_trace_examples"] += 1
            input_ids = full_ids[window_start:]
            window_labels = labels[window_start:]
            if max_supervised_tokens:
                supervised_positions = [index for index, label in enumerate(window_labels) if label != -100]
                if len(supervised_positions) > max_supervised_tokens:
                    for index in supervised_positions[:-max_supervised_tokens]:
                        window_labels[index] = -100
                    stats["supervision_capped_trace_examples"] += 1
                    stats["supervision_tokens_masked_by_cap"] += len(supervised_positions) - max_supervised_tokens
            if not any(label != -100 for label in window_labels):
                stats["skipped_no_supervised_tokens"] += 1
                continue
            examples.append(
                {
                    "input_ids": input_ids,
                    "labels": window_labels,
                    "trace_id": record.get("trace_id", f"line-{line_number}"),
                }
            )
            stats["kept_trace_examples"] += 1
    return examples, stats


class TokenDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        item = self.examples[index]
        return {"input_ids": item["input_ids"], "labels": item["labels"]}


class DynamicLengthBatchSampler(Sampler[list[int]]):
    """Pairs short full-trace examples but isolates max-context examples."""

    def __init__(self, examples: list[dict[str, Any]], threshold: int, seed: int):
        self.examples = examples
        self.threshold = threshold
        self.seed = seed

    def __iter__(self):
        order = list(range(len(self.examples)))
        random.Random(self.seed).shuffle(order)
        batches: list[list[int]] = []
        short_pending: list[int] = []
        for index in order:
            if len(self.examples[index]["input_ids"]) >= self.threshold:
                batches.append([index])
            else:
                short_pending.append(index)
                if len(short_pending) == 2:
                    batches.append(short_pending)
                    short_pending = []
        if short_pending:
            batches.append(short_pending)
        random.Random(self.seed + 1).shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        long_count = sum(len(item["input_ids"]) >= self.threshold for item in self.examples)
        short_count = len(self.examples) - long_count
        return long_count + (short_count + 1) // 2


@dataclass
class AssistantOnlyCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_size = max(len(feature["input_ids"]) for feature in features)
        input_ids, labels, attention_mask = [], [], []
        for feature in features:
            padding = max_size - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class JsonlMetricsCallback(TrainerCallback):
    """Persist every Trainer log event independently of tqdm's terminal rendering."""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "realtime_metrics.jsonl"
        # Keep the earlier portion when a job resumes from an epoch
        # checkpoint; the log is part of the experiment record.
        if not self.path.exists():
            self.path.touch()

    def on_log(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if not state.is_world_process_zero:
            return control
        record = {"created": time.time(), "step": state.global_step, **(logs or {})}
        # A long run may be moved into an organised output folder while an
        # epoch checkpoint is being saved.  Recreate the configured log
        # directory rather than losing the entire training job on that benign
        # filesystem event.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        return control


class SparseCausalLossTrainer(Trainer):
    """Causal-LM loss without materialising vocabulary logits for masked tokens.

    ``logits_to_keep`` lets assistant-only loss project only Thought/Action
    target positions.  For a short paired batch, project the union of target
    positions and then gather each row's own supervised positions.
    """

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, num_items_in_batch: Any = None):
        labels = inputs.pop("labels")
        target_mask = labels[:, 1:].ne(-100)
        positions = target_mask.any(dim=0).nonzero(as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            raise RuntimeError("Batch has no supervised next-token positions")
        # Trainer's BF16 accelerator wrapper can wrap ``forward`` in
        # ``convert_outputs_to_fp32``.  For sparse agent loss that wrapper
        # duplicates the projected [target_positions, vocab] logits (9+ GB at
        # 16K) before we immediately consume them in BF16 cross-entropy.  The
        # original forward is numerically identical here and avoids that
        # needless copy.  Non-accelerated models simply use ``forward``.
        forward = getattr(model, "_original_forward", None) or model.forward
        outputs = forward(**inputs, logits_to_keep=positions)
        row_indices, sequence_positions = target_mask.nonzero(as_tuple=True)
        projected_positions = torch.searchsorted(positions, sequence_positions)
        logits = outputs.logits[row_indices, projected_positions]
        targets = labels[:, 1:][target_mask]
        # ``logits`` is already BF16 under the training autocast context.
        # Converting a long supervised span to FP32 duplicates the complete
        # [target_positions, vocabulary] tensor (about 8 GiB at 16K here),
        # which defeats sparse projection and OOMs even at batch size one.
        # PyTorch's fused cross-entropy accepts BF16 and accumulates stably
        # without materialising that duplicate.
        ce_chunk_size = getattr(self.args, "sparse_loss_ce_chunk_size", 0)
        if ce_chunk_size and logits.shape[0] > ce_chunk_size:
            # ``cross_entropy`` needs a second vocabulary-sized workspace for
            # its log-softmax.  At 16K a single sparse span can therefore OOM
            # even though the projected BF16 logits themselves fit.  Summing
            # exact per-position losses in small chunks has the same objective
            # and gradient, but bounds that temporary workspace.
            loss_sum = None
            for start in range(0, logits.shape[0], ce_chunk_size):
                partial = F.cross_entropy(
                    logits[start : start + ce_chunk_size],
                    targets[start : start + ce_chunk_size],
                    reduction="sum",
                )
                loss_sum = partial if loss_sum is None else loss_sum + partial
            loss = loss_sum / targets.numel()
        else:
            loss = F.cross_entropy(logits, targets)
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        threshold = getattr(self.args, "dynamic_batch_under_length", 0)
        if not threshold:
            return super().get_train_dataloader()
        dataset = self.train_dataset
        if not isinstance(dataset, TokenDataset):
            raise RuntimeError("Dynamic batching expects TokenDataset.")
        sampler = DynamicLengthBatchSampler(dataset.examples, threshold, self.args.seed)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )


def install_mlp_sequence_chunking(model: Any, chunk_size: int) -> int:
    """Lower long-context activation peaks without changing MLP semantics.

    Transformer MLPs are independent across sequence positions.  Splitting
    their input on that dimension therefore produces the same output while
    avoiding a large [batch, sequence, intermediate] temporary during gradient
    checkpoint recomputation.  Attention is intentionally not changed.
    """
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ not in {"Qwen3MLP"}:
            continue
        original_forward = module.forward

        def chunked_forward(hidden_states: torch.Tensor, *args: Any, _original: Any = original_forward, **kwargs: Any) -> torch.Tensor:
            if hidden_states.ndim != 3 or hidden_states.shape[1] <= chunk_size:
                return _original(hidden_states, *args, **kwargs)
            parts = [
                _original(hidden_states[:, start : start + chunk_size], *args, **kwargs)
                for start in range(0, hidden_states.shape[1], chunk_size)
            ]
            return torch.cat(parts, dim=1)

        module.forward = chunked_forward
        patched += 1
    if not patched:
        raise RuntimeError("--mlp-sequence-chunk-size was requested but no Qwen3MLP modules were found.")
    return patched


def install_glm_parameter_lora_resume_compat(model: Any) -> None:
    """Load a GLM Parameter-LoRA checkpoint without PEFT's incompatible converter.

    Some PEFT/Transformers combinations route ``PeftModel.load_adapter`` through
    a new WeightConverter signature which cannot load the existing GLM expert
    Parameter-LoRA checkpoints.  Trainer only needs this method to restore the
    adapter before it restores optimizer/scheduler/RNG state, so load the exact
    safetensors state dict directly and leave the rest of Trainer resume intact.
    """
    def load_adapter_compat(
        checkpoint_dir: str | os.PathLike[str],
        adapter_name: str = "default",
        is_trainable: bool = False,
        **_: Any,
    ) -> tuple[list[str], list[str]]:
        if adapter_name != "default":
            raise RuntimeError(f"Unsupported adapter name while resuming GLM Parameter-LoRA: {adapter_name}")
        adapter_path = Path(checkpoint_dir) / "adapter_model.safetensors"
        if not adapter_path.is_file():
            raise RuntimeError(f"Checkpoint is missing adapter weights: {adapter_path}")
        state = load_file(str(adapter_path))
        state = {
            key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                ".lora_B.weight", ".lora_B.default.weight"
            ): value
            for key, value in state.items()
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad_missing = [key for key in missing if "lora_" in key]
        if bad_missing or unexpected:
            raise RuntimeError(
                "GLM Parameter-LoRA checkpoint does not match the configured adapter: "
                f"missing={bad_missing[:3]}, unexpected={unexpected[:3]}"
            )
        return missing, unexpected

    # Trainer invokes this instance attribute during _load_from_checkpoint.
    # The replacement changes no model/training configuration and lets Trainer
    # proceed to its normal optimizer, scheduler, RNG and step restoration.
    model.load_adapter = load_adapter_compat


def remove_completed_checkpoints(output_dir: Path) -> list[Path]:
    """Remove numeric Trainer checkpoints after the final adapter is durable.

    This function deliberately refuses to run without the final PEFT adapter
    and only removes real directories named ``checkpoint-<integer>``.  Failed
    or interrupted training never reaches this cleanup, so its resume state is
    retained.
    """
    final_adapter = output_dir / "adapter_model.safetensors"
    if not final_adapter.is_file() or final_adapter.stat().st_size == 0:
        raise RuntimeError(
            f"Refusing to delete checkpoints before a valid final adapter exists: {final_adapter}"
        )
    checkpoints = sorted(
        (
            path
            for path in output_dir.iterdir()
            if path.name.startswith("checkpoint-")
            and path.name.removeprefix("checkpoint-").isdigit()
            and path.is_dir()
            and not path.is_symlink()
        ),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)
    return checkpoints


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.right_truncate_token_ratio < 1.0:
        raise ValueError("--right-truncate-token-ratio must be in [0, 1).")
    if args.max_supervised_tokens < 0:
        raise ValueError("--max-supervised-tokens must be non-negative.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_primary_process = int(os.environ.get("RANK", "0")) == 0

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    examples, stats = build_examples(
        tokenizer,
        args.data,
        args.max_length,
        assistant_loss_mode=args.assistant_loss_mode,
        right_truncate_token_ratio=args.right_truncate_token_ratio,
        max_supervised_tokens=args.max_supervised_tokens,
    )
    if not examples:
        raise RuntimeError("No supervised assistant examples were produced; refusing to start an empty run.")
    random.Random(args.seed).shuffle(examples)
    stats.update(
        {
            "template_policy": PROTOCOL_NAME,
            "max_length": args.max_length,
            "assistant_loss_mode": args.assistant_loss_mode,
            "right_truncate_token_ratio": args.right_truncate_token_ratio,
            "max_supervised_tokens": args.max_supervised_tokens,
            "max_observed_tokens": max(len(item["input_ids"]) for item in examples),
            "supervised_tokens": sum(sum(label != -100 for label in item["labels"]) for item in examples),
        }
    )
    if is_primary_process:
        (output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2), flush=True)

    model_load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation is not None:
        model_load_kwargs["attn_implementation"] = args.attn_implementation
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    # Mistral Small 3.1 ships as the multimodal Mistral3 wrapper even for
    # text-only traces.  Its forward API is causal-LM compatible, but the
    # generic AutoModelForCausalLM registry intentionally excludes it.
    if config.__class__.__name__ == "Mistral3Config":
        from transformers.models.mistral3 import Mistral3ForConditionalGeneration
        model = Mistral3ForConditionalGeneration.from_pretrained(args.model, **model_load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_load_kwargs)
    model.config.use_cache = False
    model.enable_input_require_grads()
    target_parameters = None
    if args.glm_moe_expert_lora:
        if args.lora_dropout != 0:
            raise ValueError("--glm-moe-expert-lora requires --lora-dropout 0; Parameter-LoRA has no dropout path.")
        target_modules = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
        target_parameters = glm_moe_parameter_suffixes(model)
    else:
        target_modules = all_linear_target_suffixes(model)
    if not target_modules:
        raise RuntimeError("No linear modules found for LoRA injection.")
    print(f"LoRA target suffixes: {target_modules}", flush=True)
    if target_parameters:
        print(f"LoRA target parameter suffixes: {target_parameters}", flush=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            target_parameters=target_parameters,
        ),
    )
    if target_parameters:
        targeted = list(getattr(model, "targeted_parameter_names", []))
        print(f"LoRA targeted MoE parameters ({len(targeted)}): {targeted[:4]}", flush=True)
        required = (
            "layers.0.mlp.gate_proj.weight",
            "shared_experts.gate_proj.weight",
            "experts.gate_up_proj",
            "experts.down_proj",
            "mlp.gate.weight",
        )
        if not targeted or any(not any(name.endswith(suffix) for name in targeted) for suffix in required):
            raise RuntimeError("GLM MoE Parameter-LoRA was requested but not all expert/router parameters were wrapped.")
    model.print_trainable_parameters()
    if args.mlp_sequence_chunk_size:
        count = install_mlp_sequence_chunking(model, args.mlp_sequence_chunk_size)
        print(f"MLP sequence chunking: {count} modules, chunk_size={args.mlp_sequence_chunk_size}", flush=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        seed=args.seed,
        deepspeed=args.deepspeed_config or None,
    )
    # TrainingArguments intentionally has no experiment-specific field for a
    # dynamic sampler; attach it after validation for SparseCausalLossTrainer.
    training_args.dynamic_batch_under_length = args.dynamic_batch_under_length
    training_args.sparse_loss_ce_chunk_size = args.sparse_loss_ce_chunk_size
    trainer_cls = SparseCausalLossTrainer if args.sparse_loss else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=TokenDataset(examples),
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        callbacks=[JsonlMetricsCallback(output_dir)],
    )
    if args.resume_from_checkpoint and target_parameters:
        install_glm_parameter_lora_resume_compat(model)
        print("GLM Parameter-LoRA checkpoint compatibility loader enabled", flush=True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(output_dir))
        if not args.keep_checkpoints:
            removed = remove_completed_checkpoints(output_dir)
            print(
                f"Removed {len(removed)} completed Trainer checkpoint directories: "
                f"{[path.name for path in removed]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
