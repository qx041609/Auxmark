# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 54.7% | 37.7% |
| Cards with at least one full hit | 55.9% | 39.0% |
| Overall full-hit rate | 93 / 182 (51.1%) | 65 / 182 (35.7%) |

Card outcomes (real wins / fake wins / ties): 25 / 4 / 89. Non-tied cards: 29. One-sided exact sign-test p-value: 5.18579e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
