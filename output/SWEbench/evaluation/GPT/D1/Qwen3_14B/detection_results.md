# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 23.7% | 4.0% |
| Cards with at least one full hit | 24.8% | 4.7% |
| Overall full-hit rate | 71 / 387 (18.3%) | 16 / 387 (4.1%) |

Card outcomes (real wins / fake wins / ties): 61 / 6 / 207. Non-tied cards: 67. One-sided exact sign-test p-value: 7.47219e-13.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
