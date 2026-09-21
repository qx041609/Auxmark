# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 3.2% | 2.5% |
| Cards with at least one full hit | 5.5% | 4.8% |
| Overall full-hit rate | 10 / 379 (2.6%) | 9 / 379 (2.4%) |

Card outcomes (real wins / fake wins / ties): 5 / 4 / 136. Non-tied cards: 9. One-sided exact sign-test p-value: 0.5.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
