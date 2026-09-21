# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 37.1% | 17.8% |
| Cards with at least one full hit | 39.8% | 18.6% |
| Overall full-hit rate | 61 / 182 (33.5%) | 26 / 182 (14.3%) |

Card outcomes (real wins / fake wins / ties): 29 / 2 / 87. Non-tied cards: 31. One-sided exact sign-test p-value: 2.31434e-07.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
