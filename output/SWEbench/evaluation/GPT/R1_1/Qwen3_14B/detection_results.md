# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 27.9% | 5.8% |
| Cards with at least one full hit | 29.6% | 6.9% |
| Overall full-hit rate | 90 / 387 (23.3%) | 23 / 387 (5.9%) |

Card outcomes (real wins / fake wins / ties): 69 / 7 / 198. Non-tied cards: 76. One-sided exact sign-test p-value: 3.20898e-14.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
