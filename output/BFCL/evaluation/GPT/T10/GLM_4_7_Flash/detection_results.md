# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 50.0% | 28.4% |
| Cards with at least one full hit | 51.7% | 29.7% |
| Overall full-hit rate | 88 / 182 (48.4%) | 47 / 182 (25.8%) |

Card outcomes (real wins / fake wins / ties): 29 / 2 / 87. Non-tied cards: 31. One-sided exact sign-test p-value: 2.31434e-07.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
