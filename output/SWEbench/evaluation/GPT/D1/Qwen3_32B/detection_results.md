# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 18.7% | 2.7% |
| Cards with at least one full hit | 19.7% | 3.3% |
| Overall full-hit rate | 58 / 387 (15.0%) | 11 / 387 (2.8%) |

Card outcomes (real wins / fake wins / ties): 49 / 5 / 220. Non-tied cards: 54. One-sided exact sign-test p-value: 1.94569e-10.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
