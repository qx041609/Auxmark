# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 27.2% | 18.4% |
| Cards with at least one full hit | 28.0% | 19.2% |
| Overall full-hit rate | 41 / 146 (28.1%) | 30 / 146 (20.5%) |

Card outcomes (real wins / fake wins / ties): 13 / 2 / 110. Non-tied cards: 15. One-sided exact sign-test p-value: 0.00369263.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
