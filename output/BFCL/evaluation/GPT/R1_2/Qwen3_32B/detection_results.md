# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 55.5% | 34.7% |
| Cards with at least one full hit | 59.3% | 36.4% |
| Overall full-hit rate | 95 / 182 (52.2%) | 60 / 182 (33.0%) |

Card outcomes (real wins / fake wins / ties): 31 / 3 / 84. Non-tied cards: 34. One-sided exact sign-test p-value: 3.83006e-07.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
