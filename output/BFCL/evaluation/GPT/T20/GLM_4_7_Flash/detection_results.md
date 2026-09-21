# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 35.2% | 26.7% |
| Cards with at least one full hit | 35.6% | 27.1% |
| Overall full-hit rate | 58 / 182 (31.9%) | 43 / 182 (23.6%) |

Card outcomes (real wins / fake wins / ties): 13 / 2 / 103. Non-tied cards: 15. One-sided exact sign-test p-value: 0.00369263.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
