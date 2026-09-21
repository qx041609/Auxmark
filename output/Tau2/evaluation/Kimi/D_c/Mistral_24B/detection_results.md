# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 43.3% | 24.3% |
| Cards with at least one full hit | 49.2% | 27.9% |
| Overall full-hit rate | 68 / 192 (35.4%) | 48 / 192 (25.0%) |

Card outcomes (real wins / fake wins / ties): 37 / 14 / 71. Non-tied cards: 51. One-sided exact sign-test p-value: 0.000884599.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
