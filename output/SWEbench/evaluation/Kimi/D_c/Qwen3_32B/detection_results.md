# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 11.7% | 3.8% |
| Cards with at least one full hit | 13.7% | 4.2% |
| Overall full-hit rate | 45 / 437 (10.3%) | 14 / 437 (3.2%) |

Card outcomes (real wins / fake wins / ties): 27 / 1 / 234. Non-tied cards: 28. One-sided exact sign-test p-value: 1.08033e-07.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
