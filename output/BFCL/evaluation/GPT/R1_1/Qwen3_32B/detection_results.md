# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 54.7% | 39.4% |
| Cards with at least one full hit | 57.6% | 41.5% |
| Overall full-hit rate | 90 / 182 (49.5%) | 66 / 182 (36.3%) |

Card outcomes (real wins / fake wins / ties): 23 / 4 / 91. Non-tied cards: 27. One-sided exact sign-test p-value: 0.000155374.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
