# Released experiment artifacts

Each benchmark has the same four top-level areas:

```text
output/
├── BFCL/
│   ├── trace/
│   ├── probes/
│   ├── training/
│   └── evaluation/
├── SWEbench/                 same four areas
└── Tau2/                     same four areas
```

Within an area, data is grouped by `GPT` or `Kimi`. The release has six final S cohorts of 100 traces each and six D_c cohorts of 50 each; every D_c is selected from its corresponding S. All downstream training and probes use these frozen versions. `S/standard_traces` contains final trainable traces, `S/watermarked_traces` the agent-visible histories, `S/watermark_evidence` the Aux release/execution evidence, and `S/train.jsonl` the merged training-format copy. D_c has the same trace forms for the selected 50. SWEbench final standard traces include real tool observations; earlier collection snapshots are not included as alternative training sources.

`probes/<Teacher>/S/{real_probes,fake_probes}/` contains the final paired `probes.jsonl`, `labels.jsonl`, `pairs.jsonl`, and summaries. `training/<Teacher>/<condition>/<student>/` retains run configuration, model card, tokenizer/config where present, and training statistics; adapter weights are supplied separately. `evaluation/<Teacher>/<condition>/<student>/` retains real/fake per-probe replay results and the paired detection JSON/Markdown report. Online base-model evaluations are under `evaluation/<Teacher>/base/S/<model>/`.

GPT robustness training sets live under `trace/GPT/` with names such as `D1`, `D5`, `D10` (dilution), `T10`, `T15`, `T20` (truncation), `R1_1`, `R1_2`, `R1_3` (mixing), `D_c_Paraphrasing_attack`, and `D_c_Adaptive_attack`. Presence varies by benchmark. A trace condition without matching training or evaluation artifacts is a released dataset, not evidence that an evaluation was run. Kimi robustness artifacts are not public.

All personal machine paths in released metadata were rewritten to repository-relative `output/...` references or base-model names. Paths to external SWEbench tasks are recorded as `external_tasks/<task-id>` and require a user-provided task root for new collection. Historical checkpoint references identify where a checkpoint belonged but some referenced checkpoints are omitted with the weights. Original benchmark task text and tool observations are preserved, including task-container paths needed to interpret them.

For commands and report interpretation, start with the repository root `README.md` and `code/README.md`.
