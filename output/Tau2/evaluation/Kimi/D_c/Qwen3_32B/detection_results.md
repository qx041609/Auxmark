# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 38.5% | 23.0% |
| Cards with at least one full hit | 43.4% | 27.0% |
| Overall full-hit rate | 66 / 192 (34.4%) | 43 / 192 (22.4%) |

Card outcomes (real wins / fake wins / ties): 27 / 6 / 89. Non-tied cards: 33. One-sided exact sign-test p-value: 0.000162032.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
