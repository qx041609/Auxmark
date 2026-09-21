# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 50.4% | 39.8% |
| Cards with at least one full hit | 52.5% | 42.4% |
| Overall full-hit rate | 85 / 182 (46.7%) | 68 / 182 (37.4%) |

Card outcomes (real wins / fake wins / ties): 20 / 6 / 92. Non-tied cards: 26. One-sided exact sign-test p-value: 0.00467765.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
