# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 33.1% | 22.5% |
| Cards with at least one full hit | 33.9% | 23.7% |
| Overall full-hit rate | 51 / 182 (28.0%) | 36 / 182 (19.8%) |

Card outcomes (real wins / fake wins / ties): 16 / 4 / 98. Non-tied cards: 20. One-sided exact sign-test p-value: 0.00590897.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
