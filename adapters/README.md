# Benchmark adapters

Exactly four public Python files connect benchmark execution and robustness data construction to the core pipeline. The watermark algorithm is in `../code/watermark_proxy.py`; these scripts execute benchmark tasks, freeze final cohorts, or build additional training conditions.

| File | Benchmark and task |
|---|---|
| `bfcl.py` | BFCL multi-turn collection, in-memory function execution, S/D_c freeze. |
| `swebench.py` | SWE-bench task preparation, Docker/TerminalBench execution, S/D_c freeze. |
| `tau2.py` | Tau2 telecom user simulator and environment, S/D_c freeze. |
| `robustness.py` | Builds D/T/R training sets and runs Paraphrasing attack or Adaptive attack on D_c. |

The public command names and paths are English. All three benchmarks write under `output/<Bench>/{trace,probes,training,evaluation}/<Teacher>/`. The collector first writes `_collection_work`; `freeze --count 100` writes S, and `freeze --count 50` writes D_c. Keep the final S and D_c, which are the sources for probes and training. The `--output-root` flag redirects the same layout.

```bash
python3 adapters/bfcl.py collect --teacher GPT \
  --bfcl-root /path/to/bfcl --proxy-base-url http://127.0.0.1:8010 \
  --target-count 100
python3 adapters/bfcl.py freeze --bench BFCL --teacher GPT --count 100
python3 adapters/bfcl.py freeze --bench BFCL --teacher GPT --count 50
```

Use `adapters/swebench.py` with its TerminalBench task root and Docker environment, or `adapters/tau2.py` with Tau2 telecom data and simulator, for their corresponding benchmarks. Each adapter's `collect --help` lists its upstream input options. For SWEbench, `--tasks-root` specifies the prepared task directories; the published `candidate_manifest.json` records `external_tasks/<task-id>` rather than a machine-specific path. Give the local task root explicitly when collecting again.

## Robustness conditions

`robustness.py bfcl|swebench|tau2` builds the dilution, truncation and mixing training sets from final D_c. For example:

```bash
python3 adapters/robustness.py bfcl --teacher GPT --bfcl-root /path/to/bfcl
python3 adapters/robustness.py swebench --teacher GPT \
  --swebench-parquet /path/to/swebench-full.parquet \
  --action-tokenizer /path/to/tokenizer
python3 adapters/robustness.py tau2 --teacher GPT --tau2-root /path/to/tau2
```

The two attack subcommands create new D_c-derived training sets without modifying the original D_c:

```bash
python3 adapters/robustness.py paraphrasing-attack \
  --bench BFCL --teacher GPT --model your-model
python3 adapters/robustness.py adaptive-attack \
  --bench BFCL --teacher GPT --model your-model \
  --provider-config /path/to/private-provider-config.json
```

Paraphrasing attack rewrites assistant Thought text while preserving tool-call arguments and observations. Adaptive attack filters whole call/observation pairs and checks their pairing. The historical generated records retain their original machine-readable field names so existing data can still be parsed. `bfcl.py robustness`, `swebench.py robustness`, and `tau2.py robustness` dispatch to the same builders for compatibility.

Public robustness artifacts are for the GPT teacher only. Kimi robustness experiments are excluded. The external benchmark datasets, Docker images, service credentials, and separately distributed LoRA weights are not included here.
