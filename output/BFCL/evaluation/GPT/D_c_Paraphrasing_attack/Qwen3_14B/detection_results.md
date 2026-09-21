# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 28.4% | 17.8% |
| Cards with at least one full hit | 30.5% | 18.6% |
| Overall full-hit rate | 44 / 182 (24.2%) | 30 / 182 (16.5%) |

Card outcomes (real wins / fake wins / ties): 19 / 5 / 94. Non-tied cards: 24. One-sided exact sign-test p-value: 0.00330538.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
