# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 23.7% | 5.2% |
| Cards with at least one full hit | 25.5% | 6.6% |
| Overall full-hit rate | 72 / 387 (18.6%) | 20 / 387 (5.2%) |

Card outcomes (real wins / fake wins / ties): 58 / 6 / 210. Non-tied cards: 64. One-sided exact sign-test p-value: 4.51451e-12.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
