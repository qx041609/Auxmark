# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 58.9% | 29.3% |
| Cards with at least one full hit | 60.0% | 30.4% |
| Overall full-hit rate | 83 / 146 (56.8%) | 46 / 146 (31.5%) |

Card outcomes (real wins / fake wins / ties): 44 / 7 / 74. Non-tied cards: 51. One-sided exact sign-test p-value: 6.05763e-08.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
