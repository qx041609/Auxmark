# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 39.0% | 21.6% |
| Cards with at least one full hit | 41.5% | 24.6% |
| Overall full-hit rate | 66 / 182 (36.3%) | 38 / 182 (20.9%) |

Card outcomes (real wins / fake wins / ties): 28 / 6 / 84. Non-tied cards: 34. One-sided exact sign-test p-value: 9.75628e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
