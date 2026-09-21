# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 19.1% | 5.6% |
| Cards with at least one full hit | 20.8% | 6.6% |
| Overall full-hit rate | 58 / 387 (15.0%) | 23 / 387 (5.9%) |

Card outcomes (real wins / fake wins / ties): 45 / 9 / 220. Non-tied cards: 54. One-sided exact sign-test p-value: 3.64422e-07.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
