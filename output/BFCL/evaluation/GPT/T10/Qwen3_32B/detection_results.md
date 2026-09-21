# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 40.3% | 25.0% |
| Cards with at least one full hit | 44.9% | 26.3% |
| Overall full-hit rate | 66 / 182 (36.3%) | 40 / 182 (22.0%) |

Card outcomes (real wins / fake wins / ties): 26 / 6 / 86. Non-tied cards: 32. One-sided exact sign-test p-value: 0.000267526.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
