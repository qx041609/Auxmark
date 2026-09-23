# Auxmark — anti-distillation watermark for agent traces

AuxMark is a behavioral watermarking framework designed to trace and defend against unauthorized agent model distillation. The system works by dynamically inserting safe, non-essential "auxiliary actions" into the interaction trajectories of the teacher agent and saving the associated contexts as private evidence cards. When auditing a suspicious student model, AuxMark utilizes this private evidence to construct paired real and fake probes, verifying whether the student model has retained these watermarked behaviors through a card-level sign test. An optional trace-ranking step performs attribution analysis on candidate source traces after probe replay.
<img width="511" height="488" alt="屏幕截图 2026-09-23 192423" src="https://github.com/user-attachments/assets/f267be48-3273-47eb-aa68-27e08339f781" />



## Install

Use Python 3.11 or later for the core scripts and install the direct Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Training and local replay also need a CUDA-compatible PyTorch installation and enough GPU memory for the selected base model. Install the matching PyTorch build for your system before the command above if necessary. The three collection adapters additionally need their upstream benchmark assets and runtimes:

| Adapter | External input/runtime |
|---|---|
| `adapters/bfcl.py` | BFCL multi-turn data, function schemas, and its execution environment (`--bfcl-root`). |
| `adapters/swebench.py` | SWE-bench task data and a TerminalBench-compatible Docker task runner (`--terminalbench-root`, optionally `--tasks-root`). |
| `adapters/tau2.py` | Tau2 source checkout, telecom data and simulator (`--tau2-root`), plus its own dependencies. |

Install those benchmark packages from their upstream repositories in the environment used for collection. The released trace/probe/results files and trace ranking do not require benchmark execution environments. This requirements file covers the Python imports in this repository; provider clients and benchmark packages may impose additional version constraints.

## Quickstart: inspect existing results

Each evaluation directory contains `detection_results.json` and a human-readable `detection_results.md`. For example:

```bash
cat output/BFCL/evaluation/GPT/D_c/GLM_4_7_Flash/detection_results.md
```

To rank the 100 source traces for a saved student replay, using the known 50 training traces only to compute Precision@K:

```bash
python3 code/rank_traces.py \
  --bench BFCL --teacher GPT --condition D_c \
  --model GLM_4_7_Flash --known-dc
```

This reads saved probe labels and replay `results.jsonl`, and writes `trace_ranking/trace_ranking.jsonl` and `ranking_summary.json` under that evaluation directory. It makes no model request. Omit `--known-dc` when the training subset is unknown.

## Full workflow

Run commands from the repository root. Replace `BFCL` and `GPT` with another benchmark and teacher cohort as needed. The same layout is used for all three benchmarks.

### 1. Embed and collect traces

```bash
export AGENTWM_TEACHER_API_KEY='your-api-key'
export AGENTWM_AUX_API_KEY='your-aux-api-key'
export AGENTWM_SECRET_SEED='your-private-watermark-seed'
python3 code/embedding.py serve \
  --bench BFCL --teacher GPT \
  --teacher-url https://your-provider.example/v1 \
  --teacher-model your-model --filter-model your-review-model \
  --base-probability 0.1 --probability-increment 0.1 \
  --max-probability 1.0 --total-budget 10 --port 8010
```

`embedding.py serve` is the only service that needs to be started; it loads `watermark_proxy.py` internally, so do not start the latter separately. AuxMark exposes the processed, watermarked teacher as a standalone OpenAI-compatible endpoint: an agent or benchmark adapter only needs to direct its model requests to this endpoint, without embedding AuxMark into the agent framework itself. The local endpoint accepts OpenAI-style `POST /v1/chat/completions`. A benchmark adapter sends messages and the tool schema, executes the returned tool call, and sends the real observation in the next turn. On the first request, the endpoint checks its in-memory cache, `--safety-path`, and the current run's `safety_policies/` directory; only a cache miss calls the safety model, and the generated policy is saved for reuse. The same policy is used at every step of that trace, and concurrent traces with the same tool schema share one generation. The proxy then chooses a normal Core action, evaluates safe auxiliary (Aux) candidates, applies its probability schedule and budget, and records the agent-visible history and release evidence. The seed, credentials, safety policies, and watermark evidence must be kept private.

Collection uses `output/<Bench>/trace/<Teacher>/_collection_work/`. For example, to collect and freeze BFCL:

```bash
python3 adapters/bfcl.py collect \
  --teacher GPT --bfcl-root /path/to/bfcl \
  --proxy-base-url http://127.0.0.1:8010 --target-count 100
python3 adapters/bfcl.py freeze --bench BFCL --teacher GPT --count 100
python3 adapters/bfcl.py freeze --bench BFCL --teacher GPT --count 50
```

`freeze` produces `S` (100 final candidate traces) and `D_c` (50 traces selected from S for training). Use the matching `collect`/`freeze` subcommands in `adapters/swebench.py` and `adapters/tau2.py` for those benchmarks. Frozen `standard_traces` and `train.jsonl` are the versions used downstream. `watermarked_traces` and `watermark_evidence` retain the agent-visible history and private release evidence needed for probe construction. The released SWEbench traces include the finalized real tool observations; the earlier collection snapshots are not training inputs.

### 2. Train a student (or use separately supplied weights)

```bash
python3 code/distill.py \
  --bench BFCL --teacher GPT --condition D_c \
  --run-name GLM_4_7_Flash \
  --model /path/to/base-model
```

This reads `output/BFCL/trace/GPT/D_c/train.jsonl` and writes LoRA output under `output/BFCL/training/GPT/D_c/GLM_4_7_Flash/`. A complete trace is one sample. System, user, and tool messages provide context; assistant text and tool calls receive loss. Training and detection share the text renderer in `code/detect.py`. Defaults include BF16, 16K maximum length, 20 epochs, LoRA rank 32/alpha 64, batch size 1, and gradient accumulation 2; inspect `python3 code/distill.py --help` and the recorded `run_config.json` before reproducing a particular run. `--moe` activates the existing GLM expert Parameter-LoRA branch.

The public training directories contain configuration and statistics, but **no adapter weights**. For an already trained student, skip this step and supply the separately distributed adapter path when running detection. `training_args.bin` is a serialized Trainer configuration, not a model weight; it is not loaded by detection and is omitted from this release. Some historic `resume_from_checkpoint` fields name checkpoints that are likewise not shipped, so resuming those exact runs requires the original checkpoint files.

### 3. Generate paired probes

```bash
python3 code/generate_probes.py \
  --bench BFCL --teacher GPT --model your-model \
  --workers 4 --variants-per-card 3
```

This reads the final S `standard_traces` and `watermark_evidence`, then writes `probes/<Teacher>/S/real_probes/` and `fake_probes/`. Each side has `probes.jsonl`, `labels.jsonl`, and pairing metadata. Real probes follow the original Core path with parameter variants; fake probes use a rewritten Core mechanism targeting the same Aux behavior. Existing released probe files can be replayed directly, so probe generation is optional for verification.

### 4. Detect on the already trained adapter

```bash
python3 code/detect.py \
  --bench BFCL --teacher GPT --condition D_c \
  --run-name GLM_4_7_Flash --mode local \
  --base-model /path/to/base-model --adapter /path/to/adapter \
  --batch-size 4
```

`--adapter` must point to a directory containing `adapter_model.safetensors` and a compatible `adapter_config.json`. If the configuration is distributed separately, pass its directory with `--adapter-config-dir`. For an online OpenAI-compatible target:

```bash
export OPENROUTER_API_KEY='your-api-key'
python3 code/detect.py \
  --bench BFCL --teacher GPT --condition D_c \
  --run-name target-model --mode online \
  --model your-model --endpoint https://your-provider.example/v1/chat/completions
```

The detector replays both probe sides against the **same** target, parses text-form tool calls, and writes each side's `results.jsonl` plus `detection_results.json` and `.md`. It does not send structured tools to the online endpoint. `--report-only` recomputes the paired statistics from existing side results without model requests, but the current CLI still requires the relevant `--mode` and model argument. Explicit `--real-probes`, `--real-labels`, `--fake-probes`, `--fake-labels`, and `--out-dir` override automatic benchmark paths.

### 5. Rank candidate source traces

Run the quickstart ranking command after detection. For each source trace, the ranker combines group percentile ranks of real/fake full-hit and tool-only rates:

`score = Q(real_full) - 0.25 Q(fake_full) + 0.10 Q(real_tool_only) - 0.10 Q(fake_tool_only)`.

Only traces with matched probe pairs are ranked. Known D_c labels are used solely for offline Precision@K, never for the score.

## Reading detection results

| Field | Meaning |
|---|---|
| `real_card_mean_full_hit`, `fake_card_mean_full_hit` | Mean full-hit fraction across release cards on each side. |
| `real_full_hit_rate`, `fake_full_hit_rate` | Probe-level full-hit rates over paired probes. |
| `real_wins`, `fake_wins`, `ties` | Number of release cards where one side has more full hits, or both tie. |
| `p_one_sided_exact_sign` | One-sided exact sign-test tail on non-tied cards. |

A card is the unit of the sign test: variants from the same release are **not** independent samples. The report is evidence about the tested trace/probe cohort and target model. It does not make an account-level claim. Trace identification is a separate ranking over the S candidates. See `code/README.md` for input and scoring details.

## Directory layout

```text
Auxmark/
├── code/                  embedding, training, probe, detection, ranking, helpers
├── adapters/              bfcl.py, swebench.py, tau2.py, robustness.py
├── output/
│   ├── BFCL/
│   ├── SWEbench/
│   └── Tau2/
│       ├── trace/<Teacher>/S|D_c|<robustness condition>/
│       ├── probes/<Teacher>/S/real_probes|fake_probes/
│       ├── training/<Teacher>/<condition>/<student>/
│       └── evaluation/<Teacher>/<condition>/<student>/
└── requirements.txt
```

The four area directories exist **under each benchmark**. `<Teacher>` is `GPT` or `Kimi`. Public Kimi material is limited to S, D_c, probes, and baseline training/evaluation; Kimi robustness runs are excluded. GPT robustness conditions include dilution (`D1/D5/D10`), truncation (`T10/T15/T20`), mixing (`R1_1/R1_2/R1_3`), `D_c_Paraphrasing_attack`, and `D_c_Adaptive_attack`, where present. The attacks build separate training sets from D_c. See `output/README.md` for exact contents and known gaps.

## Configuration and troubleshooting

- **Credentials:** pass provider keys through environment variables or `--key-file`; never commit a real key. `model_config.py` also supports optional private provider JSON under `~/.config/agentwm/`. For embedding, use `AGENTWM_TEACHER_API_KEY`, `AGENTWM_AUX_API_KEY`, and `AGENTWM_SECRET_SEED`; online detection uses `OPENROUTER_API_KEY` or `--key-file`.
- **Base model:** published `adapter_config.json` records only the base model's name, not the old machine path. Supply the matching locally installed base model with `--base-model`. An adapter from a different architecture or tokenizer will not load correctly.
- **Historical metadata:** paths in public configs are repository-relative (`output/...`). Treat `run_config.json` as a record of the original settings, not an executable command line. External checkpoint, benchmark, and DeepSpeed paths may require local replacements.
- **Missing probe or result side:** both real and fake `results.jsonl` are required for the paired report.
- **Provider errors or low coverage:** check model ID, endpoint, key, tool-call text format, and each side's replay `results.jsonl`. A failed or truncated request cannot be repaired by changing the aggregate report.
- **Collection mismatch:** freeze S before D_c, and keep the same bench/teacher combination throughout. Do not mix probes or weights from different conditions.

The release changes path layout, names, and personal-path metadata. Core trace samples, probe/label rows, training data, and saved detection result JSON remain byte-identical to the collected versions. Original benchmark task payloads can contain container paths such as `/root/...`; those strings are part of the tasks and are intentionally preserved.
