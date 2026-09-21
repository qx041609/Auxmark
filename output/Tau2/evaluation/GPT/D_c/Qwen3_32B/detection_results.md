# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 61.1% | 22.0% |
| Cards with at least one full hit | 62.4% | 23.2% |
| Overall full-hit rate | 85 / 146 (58.2%) | 36 / 146 (24.7%) |

Card outcomes (real wins / fake wins / ties): 57 / 7 / 61. Non-tied cards: 64. One-sided exact sign-test p-value: 3.81907e-11.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
