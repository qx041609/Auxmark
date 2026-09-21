# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 3.1% | 0.9% |
| Cards with at least one full hit | 3.4% | 1.9% |
| Overall full-hit rate | 12 / 437 (2.7%) | 5 / 437 (1.1%) |

Card outcomes (real wins / fake wins / ties): 8 / 3 / 251. Non-tied cards: 11. One-sided exact sign-test p-value: 0.113281.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
