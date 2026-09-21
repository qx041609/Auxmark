# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 42.4% | 33.5% |
| Cards with at least one full hit | 44.1% | 34.7% |
| Overall full-hit rate | 73 / 182 (40.1%) | 55 / 182 (30.2%) |

Card outcomes (real wins / fake wins / ties): 15 / 2 / 101. Non-tied cards: 17. One-sided exact sign-test p-value: 0.00117493.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
