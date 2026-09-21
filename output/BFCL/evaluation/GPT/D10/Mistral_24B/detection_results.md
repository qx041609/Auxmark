# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 58.1% | 44.1% |
| Cards with at least one full hit | 58.5% | 45.8% |
| Overall full-hit rate | 101 / 182 (55.5%) | 73 / 182 (40.1%) |

Card outcomes (real wins / fake wins / ties): 20 / 2 / 96. Non-tied cards: 22. One-sided exact sign-test p-value: 6.05583e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
