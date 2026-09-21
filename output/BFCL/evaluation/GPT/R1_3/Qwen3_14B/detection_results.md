# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 48.7% | 32.2% |
| Cards with at least one full hit | 50.8% | 34.7% |
| Overall full-hit rate | 79 / 182 (43.4%) | 55 / 182 (30.2%) |

Card outcomes (real wins / fake wins / ties): 26 / 5 / 87. Non-tied cards: 31. One-sided exact sign-test p-value: 9.60976e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
