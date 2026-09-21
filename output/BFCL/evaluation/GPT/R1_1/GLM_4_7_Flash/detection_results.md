# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 58.9% | 39.8% |
| Cards with at least one full hit | 61.0% | 41.5% |
| Overall full-hit rate | 102 / 182 (56.0%) | 68 / 182 (37.4%) |

Card outcomes (real wins / fake wins / ties): 27 / 3 / 88. Non-tied cards: 30. One-sided exact sign-test p-value: 4.21517e-06.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
