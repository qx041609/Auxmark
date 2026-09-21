# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 51.3% | 33.9% |
| Cards with at least one full hit | 54.2% | 34.7% |
| Overall full-hit rate | 86 / 182 (47.3%) | 52 / 182 (28.6%) |

Card outcomes (real wins / fake wins / ties): 25 / 3 / 90. Non-tied cards: 28. One-sided exact sign-test p-value: 1.37202e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
