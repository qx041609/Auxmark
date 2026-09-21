# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 13.7% | 5.7% |
| Cards with at least one full hit | 15.7% | 7.3% |
| Overall full-hit rate | 51 / 387 (13.2%) | 23 / 387 (5.9%) |

Card outcomes (real wins / fake wins / ties): 34 / 11 / 229. Non-tied cards: 45. One-sided exact sign-test p-value: 0.000412041.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
