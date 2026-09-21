# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 35.5% | 8.2% |
| Cards with at least one full hit | 37.2% | 9.5% |
| Overall full-hit rate | 108 / 387 (27.9%) | 32 / 387 (8.3%) |

Card outcomes (real wins / fake wins / ties): 81 / 5 / 188. Non-tied cards: 86. One-sided exact sign-test p-value: 4.78937e-19.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
