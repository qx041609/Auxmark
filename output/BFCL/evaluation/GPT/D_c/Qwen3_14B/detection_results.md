# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 30.9% | 18.6% |
| Cards with at least one full hit | 32.2% | 19.5% |
| Overall full-hit rate | 44 / 182 (24.2%) | 28 / 182 (15.4%) |

Card outcomes (real wins / fake wins / ties): 20 / 4 / 94. Non-tied cards: 24. One-sided exact sign-test p-value: 0.00077194.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
