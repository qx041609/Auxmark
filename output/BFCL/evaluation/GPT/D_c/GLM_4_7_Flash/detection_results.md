# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 61.9% | 47.9% |
| Cards with at least one full hit | 63.6% | 50.0% |
| Overall full-hit rate | 108 / 182 (59.3%) | 80 / 182 (44.0%) |

Card outcomes (real wins / fake wins / ties): 19 / 1 / 98. Non-tied cards: 20. One-sided exact sign-test p-value: 2.00272e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
