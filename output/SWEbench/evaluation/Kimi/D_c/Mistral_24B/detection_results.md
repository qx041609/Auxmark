# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 23.2% | 11.8% |
| Cards with at least one full hit | 24.8% | 14.1% |
| Overall full-hit rate | 85 / 437 (19.5%) | 50 / 437 (11.4%) |

Card outcomes (real wins / fake wins / ties): 38 / 9 / 215. Non-tied cards: 47. One-sided exact sign-test p-value: 1.2452e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
