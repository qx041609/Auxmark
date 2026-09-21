# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 27.9% | 5.7% |
| Cards with at least one full hit | 29.2% | 6.6% |
| Overall full-hit rate | 81 / 387 (20.9%) | 21 / 387 (5.4%) |

Card outcomes (real wins / fake wins / ties): 67 / 6 / 201. Non-tied cards: 73. One-sided exact sign-test p-value: 1.97363e-14.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
