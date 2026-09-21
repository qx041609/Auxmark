# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 43.6% | 28.8% |
| Cards with at least one full hit | 47.5% | 30.5% |
| Overall full-hit rate | 75 / 182 (41.2%) | 48 / 182 (26.4%) |

Card outcomes (real wins / fake wins / ties): 27 / 7 / 84. Non-tied cards: 34. One-sided exact sign-test p-value: 0.000410698.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
