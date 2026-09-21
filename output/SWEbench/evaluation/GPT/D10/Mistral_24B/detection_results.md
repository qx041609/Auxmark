# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 34.8% | 7.6% |
| Cards with at least one full hit | 36.5% | 8.4% |
| Overall full-hit rate | 106 / 387 (27.4%) | 27 / 387 (7.0%) |

Card outcomes (real wins / fake wins / ties): 81 / 5 / 188. Non-tied cards: 86. One-sided exact sign-test p-value: 4.78937e-19.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
