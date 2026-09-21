# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 23.7% | 6.2% |
| Cards with at least one full hit | 24.5% | 7.3% |
| Overall full-hit rate | 72 / 387 (18.6%) | 24 / 387 (6.2%) |

Card outcomes (real wins / fake wins / ties): 55 / 7 / 212. Non-tied cards: 62. One-sided exact sign-test p-value: 1.21504e-10.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
