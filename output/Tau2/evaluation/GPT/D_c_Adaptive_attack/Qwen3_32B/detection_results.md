# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 52.8% | 15.6% |
| Cards with at least one full hit | 53.6% | 16.8% |
| Overall full-hit rate | 69 / 146 (47.3%) | 21 / 146 (14.4%) |

Card outcomes (real wins / fake wins / ties): 52 / 4 / 69. Non-tied cards: 56. One-sided exact sign-test p-value: 5.50403e-12.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
