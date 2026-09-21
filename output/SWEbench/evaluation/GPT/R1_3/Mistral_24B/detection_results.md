# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 37.1% | 6.9% |
| Cards with at least one full hit | 38.3% | 8.8% |
| Overall full-hit rate | 114 / 387 (29.5%) | 27 / 387 (7.0%) |

Card outcomes (real wins / fake wins / ties): 90 / 6 / 178. Non-tied cards: 96. One-sided exact sign-test p-value: 1.25163e-20.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
