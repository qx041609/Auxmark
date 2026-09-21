# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 56.8% | 39.0% |
| Cards with at least one full hit | 59.3% | 39.8% |
| Overall full-hit rate | 99 / 182 (54.4%) | 65 / 182 (35.7%) |

Card outcomes (real wins / fake wins / ties): 26 / 2 / 90. Non-tied cards: 28. One-sided exact sign-test p-value: 1.51619e-06.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
