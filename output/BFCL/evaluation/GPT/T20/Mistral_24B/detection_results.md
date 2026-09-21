# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 39.8% | 30.5% |
| Cards with at least one full hit | 41.5% | 32.2% |
| Overall full-hit rate | 67 / 182 (36.8%) | 48 / 182 (26.4%) |

Card outcomes (real wins / fake wins / ties): 14 / 2 / 102. Non-tied cards: 16. One-sided exact sign-test p-value: 0.00209045.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
