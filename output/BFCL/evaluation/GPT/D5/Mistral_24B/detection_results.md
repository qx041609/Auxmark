# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 53.8% | 44.1% |
| Cards with at least one full hit | 54.2% | 46.6% |
| Overall full-hit rate | 94 / 182 (51.6%) | 76 / 182 (41.8%) |

Card outcomes (real wins / fake wins / ties): 17 / 5 / 96. Non-tied cards: 22. One-sided exact sign-test p-value: 0.00845027.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
