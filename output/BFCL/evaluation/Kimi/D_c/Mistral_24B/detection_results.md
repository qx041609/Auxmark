# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 47.0% | 29.0% |
| Cards with at least one full hit | 55.2% | 36.6% |
| Overall full-hit rate | 175 / 379 (46.2%) | 105 / 379 (27.7%) |

Card outcomes (real wins / fake wins / ties): 45 / 10 / 90. Non-tied cards: 55. One-sided exact sign-test p-value: 1.02863e-06.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
