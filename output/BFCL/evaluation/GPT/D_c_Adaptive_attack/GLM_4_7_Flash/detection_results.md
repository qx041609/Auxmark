# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 56.4% | 44.5% |
| Cards with at least one full hit | 58.5% | 45.8% |
| Overall full-hit rate | 100 / 182 (54.9%) | 76 / 182 (41.8%) |

Card outcomes (real wins / fake wins / ties): 16 / 1 / 101. Non-tied cards: 17. One-sided exact sign-test p-value: 0.000137329.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
