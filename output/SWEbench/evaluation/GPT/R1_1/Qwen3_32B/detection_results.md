# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 18.6% | 2.6% |
| Cards with at least one full hit | 19.0% | 3.3% |
| Overall full-hit rate | 53 / 387 (13.7%) | 9 / 387 (2.3%) |

Card outcomes (real wins / fake wins / ties): 49 / 6 / 219. Non-tied cards: 55. One-sided exact sign-test p-value: 9.11417e-10.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
