# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 65.7% | 49.6% |
| Cards with at least one full hit | 67.8% | 50.8% |
| Overall full-hit rate | 115 / 182 (63.2%) | 83 / 182 (45.6%) |

Card outcomes (real wins / fake wins / ties): 24 / 2 / 92. Non-tied cards: 26. One-sided exact sign-test p-value: 5.24521e-06.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
