# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 54.2% | 35.0% |
| Cards with at least one full hit | 58.5% | 38.1% |
| Overall full-hit rate | 89 / 182 (48.9%) | 58 / 182 (31.9%) |

Card outcomes (real wins / fake wins / ties): 30 / 9 / 79. Non-tied cards: 39. One-sided exact sign-test p-value: 0.00053251.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
