# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 19.8% | 4.4% |
| Cards with at least one full hit | 21.2% | 5.8% |
| Overall full-hit rate | 63 / 387 (16.3%) | 18 / 387 (4.7%) |

Card outcomes (real wins / fake wins / ties): 50 / 6 / 218. Non-tied cards: 56. One-sided exact sign-test p-value: 5.09105e-10.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
