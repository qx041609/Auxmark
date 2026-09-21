# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 59.3% | 43.6% |
| Cards with at least one full hit | 61.0% | 45.8% |
| Overall full-hit rate | 102 / 182 (56.0%) | 76 / 182 (41.8%) |

Card outcomes (real wins / fake wins / ties): 24 / 3 / 91. Non-tied cards: 27. One-sided exact sign-test p-value: 2.46167e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
