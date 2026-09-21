# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 57.2% | 42.4% |
| Cards with at least one full hit | 60.2% | 44.1% |
| Overall full-hit rate | 100 / 182 (54.9%) | 74 / 182 (40.7%) |

Card outcomes (real wins / fake wins / ties): 25 / 7 / 86. Non-tied cards: 32. One-sided exact sign-test p-value: 0.0010512.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
