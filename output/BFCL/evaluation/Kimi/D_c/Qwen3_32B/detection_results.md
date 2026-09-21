# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 22.5% | 6.4% |
| Cards with at least one full hit | 29.7% | 9.7% |
| Overall full-hit rate | 82 / 379 (21.6%) | 18 / 379 (4.7%) |

Card outcomes (real wins / fake wins / ties): 39 / 5 / 101. Non-tied cards: 44. One-sided exact sign-test p-value: 7.02581e-08.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
