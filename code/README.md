# Core scripts

Run from the repository root. Every entry point accepts explicit paths; `--bench BFCL|SWEbench|Tau2 --teacher GPT|Kimi` selects the corresponding path below `output/` automatically.

| File | Role |
|---|---|
| `embedding.py` | Starts the OpenAI-compatible teacher proxy, selects safe Aux actions, and records complete traces and watermark evidence. |
| `watermark_proxy.py` | The watermark engine: scheduling, candidate checks, argument binding, release and execution records. Imported by `embedding.py`. |
| `distill.py` | LoRA distillation from finalized `trace/<Teacher>/<condition>/train.jsonl`. |
| `generate_probes.py` | Creates paired real/fake probes from final S traces and release evidence. |
| `detect.py` | Shared text protocol, local or online probe replay, strict tool/full-hit scoring, card-level sign test. |
| `rank_traces.py` | Ranks source traces from saved probe labels and replay results; optional Precision@K with known D_c. |
| `model_config.py` | Provider key and base-URL resolution. |
| `output_layout.py` | Canonical bench/teacher path selection. |

## Shared data format

Traces use OpenAI-style `messages`, assistant `tool_calls`, tool observations, and the tool schema. `standard_traces/*.json` and `train.jsonl` are the final training-ready versions. `watermarked_traces` is the history visible to the agent. `watermark_evidence` contains Aux release/execution records and the mapping between internal and visible call IDs. Keep this evidence private if you generate a new watermark; the included release evidence is needed to reproduce the published probes.

The text renderer lives in `detect.py` and is imported by `distill.py` and the SWEbench robustness builder. It serializes the tool schema into the system text and the conversation into a stable user text. Online replay requests text output and parses `<tool_call>`; it does not use the provider's structured tool API. The existing regexes that recognize Chinese watermark terms and quota/balance text are algorithmic input handling, not comments or filenames.

## Commands

```bash
# Embed; connect one of the benchmark collection adapters to this endpoint.
python3 code/embedding.py serve --bench BFCL --teacher GPT \
  --teacher-url https://your-provider.example/v1 \
  --teacher-model your-model --filter-model your-review-model \
  --secret-seed your-private-seed --port 8010

# Train on the 50 D_c traces. Use a local path or matching model identifier.
python3 code/distill.py --bench BFCL --teacher GPT --condition D_c \
  --run-name GLM_4_7_Flash --model /path/to/base-model

# Regenerate probes from the 100 final S traces.
python3 code/generate_probes.py --bench BFCL --teacher GPT --model your-model

# Replay existing probes against a separately supplied trained adapter.
python3 code/detect.py --bench BFCL --teacher GPT --condition D_c \
  --run-name GLM_4_7_Flash --mode local \
  --base-model /path/to/base-model --adapter /path/to/adapter

# Rank saved replay results. D_c contributes labels to Precision@K only.
python3 code/rank_traces.py --bench BFCL --teacher GPT \
  --condition D_c --model GLM_4_7_Flash --known-dc
```

To evaluate an online target, use `detect.py --mode online --model <provider-model-id> --endpoint <chat-completions-url>`. `OPENROUTER_API_KEY` or `--key-file` supplies its key. `--report-only` rebuilds a paired report from existing real/fake result files without hitting the target. Check each entry point's `--help` for optional overrides and resource controls.

## Detection statistics

Each probe has a paired label. Tool-hit requires the expected tool; full-hit additionally checks the tool arguments against the label's strict matching rule. Probe variants from the same Aux release are grouped into one card. The report compares each card's real and fake full-hit counts, records real wins, fake wins and ties, and computes the one-sided exact sign-test p-value from non-tied cards. It also reports per-card and per-probe hit rates. The report is a trace-cohort test; the optional ranker identifies which of the 100 source traces contributed more evidence.

The ranker computes group percentile ranks for each candidate source trace. Its frozen score is `Q(real_full) - 0.25 Q(fake_full) + 0.10 Q(real_tool_only) - 0.10 Q(fake_tool_only)`. Ties use lower fake full-hit rate, higher real full-hit rate, then trace ID. A source must have matched pairs to be ranked. Known D_c labels are never used to compute the score.

## Files written

| Command | Output |
|---|---|
| `embedding.py serve` | `trace/<Teacher>/_collection_work/embedding/{standard_traces,watermarked_traces,watermark_evidence,safety_policies}/` |
| `distill.py` | `training/<Teacher>/<condition>/<student>/` |
| `generate_probes.py` | `probes/<Teacher>/S/{real_probes,fake_probes}/` |
| `detect.py` | `evaluation/<Teacher>/<condition>/<student>/{real_probes,fake_probes}/results.jsonl`, `detection_results.json`, `.md` |
| `rank_traces.py` | `evaluation/<Teacher>/<condition>/<student>/trace_ranking/{trace_ranking.jsonl,ranking_summary.json}` |

The public package excludes LoRA tensors and `training_args.bin`. Detection requires the compatible LoRA weights and `adapter_config.json`, plus the matching base model; it does not read `training_args.bin`. Historical training metadata is retained for parameter inspection. See the root README for dependency and benchmark setup details.
