# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 39.8% | 28.4% |
| Cards with at least one full hit | 41.5% | 29.7% |
| Overall full-hit rate | 64 / 182 (35.2%) | 47 / 182 (25.8%) |

Card outcomes (real wins / fake wins / ties): 20 / 7 / 91. Non-tied cards: 27. One-sided exact sign-test p-value: 0.00957865.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
