# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 22.9% | 2.7% |
| Cards with at least one full hit | 23.7% | 3.6% |
| Overall full-hit rate | 68 / 387 (17.6%) | 10 / 387 (2.6%) |

Card outcomes (real wins / fake wins / ties): 61 / 6 / 207. Non-tied cards: 67. One-sided exact sign-test p-value: 7.47219e-13.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
