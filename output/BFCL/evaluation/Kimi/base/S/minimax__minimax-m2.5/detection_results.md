# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 1.4% | 0.7% |
| Cards with at least one full hit | 3.4% | 2.1% |
| Overall full-hit rate | 6 / 379 (1.6%) | 3 / 379 (0.8%) |

Card outcomes (real wins / fake wins / ties): 4 / 2 / 139. Non-tied cards: 6. One-sided exact sign-test p-value: 0.34375.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
