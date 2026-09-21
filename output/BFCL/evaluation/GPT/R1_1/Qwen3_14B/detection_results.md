# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 55.5% | 34.3% |
| Cards with at least one full hit | 56.8% | 36.4% |
| Overall full-hit rate | 94 / 182 (51.6%) | 58 / 182 (31.9%) |

Card outcomes (real wins / fake wins / ties): 29 / 6 / 83. Non-tied cards: 35. One-sided exact sign-test p-value: 5.84209e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
