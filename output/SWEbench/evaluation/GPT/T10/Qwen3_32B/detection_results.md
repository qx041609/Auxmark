# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 19.3% | 5.6% |
| Cards with at least one full hit | 21.2% | 7.3% |
| Overall full-hit rate | 62 / 387 (16.0%) | 22 / 387 (5.7%) |

Card outcomes (real wins / fake wins / ties): 47 / 8 / 219. Non-tied cards: 55. One-sided exact sign-test p-value: 4.0338e-08.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
