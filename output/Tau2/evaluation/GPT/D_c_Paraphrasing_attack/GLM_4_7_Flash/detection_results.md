# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 31.5% | 16.0% |
| Cards with at least one full hit | 33.6% | 16.8% |
| Overall full-hit rate | 43 / 146 (29.5%) | 21 / 146 (14.4%) |

Card outcomes (real wins / fake wins / ties): 25 / 3 / 97. Non-tied cards: 28. One-sided exact sign-test p-value: 1.37202e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
