# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 55.1% | 36.0% |
| Cards with at least one full hit | 57.6% | 39.0% |
| Overall full-hit rate | 91 / 182 (50.0%) | 61 / 182 (33.5%) |

Card outcomes (real wins / fake wins / ties): 28 / 4 / 86. Non-tied cards: 32. One-sided exact sign-test p-value: 9.6506e-06.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
