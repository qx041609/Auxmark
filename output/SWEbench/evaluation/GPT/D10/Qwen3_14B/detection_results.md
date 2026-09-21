# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 25.3% | 4.7% |
| Cards with at least one full hit | 27.0% | 5.5% |
| Overall full-hit rate | 76 / 387 (19.6%) | 17 / 387 (4.4%) |

Card outcomes (real wins / fake wins / ties): 62 / 5 / 207. Non-tied cards: 67. One-sided exact sign-test p-value: 7.09767e-14.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
