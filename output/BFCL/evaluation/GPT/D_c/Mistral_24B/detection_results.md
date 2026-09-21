# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 54.7% | 36.0% |
| Cards with at least one full hit | 55.1% | 38.1% |
| Overall full-hit rate | 93 / 182 (51.1%) | 59 / 182 (32.4%) |

Card outcomes (real wins / fake wins / ties): 25 / 1 / 92. Non-tied cards: 26. One-sided exact sign-test p-value: 4.02331e-07.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
