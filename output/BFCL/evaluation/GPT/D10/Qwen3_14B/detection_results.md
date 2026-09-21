# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 53.8% | 37.3% |
| Cards with at least one full hit | 55.1% | 39.8% |
| Overall full-hit rate | 90 / 182 (49.5%) | 62 / 182 (34.1%) |

Card outcomes (real wins / fake wins / ties): 25 / 2 / 91. Non-tied cards: 27. One-sided exact sign-test p-value: 2.82377e-06.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
