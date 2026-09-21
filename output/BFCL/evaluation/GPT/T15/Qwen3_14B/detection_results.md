# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 26.7% | 16.9% |
| Cards with at least one full hit | 27.1% | 17.8% |
| Overall full-hit rate | 41 / 182 (22.5%) | 25 / 182 (13.7%) |

Card outcomes (real wins / fake wins / ties): 16 / 4 / 98. Non-tied cards: 20. One-sided exact sign-test p-value: 0.00590897.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
