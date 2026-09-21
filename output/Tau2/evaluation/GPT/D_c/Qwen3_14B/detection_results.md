# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 60.0% | 27.3% |
| Cards with at least one full hit | 60.0% | 28.8% |
| Overall full-hit rate | 86 / 146 (58.9%) | 44 / 146 (30.1%) |

Card outcomes (real wins / fake wins / ties): 48 / 6 / 71. Non-tied cards: 54. One-sided exact sign-test p-value: 1.62827e-09.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
