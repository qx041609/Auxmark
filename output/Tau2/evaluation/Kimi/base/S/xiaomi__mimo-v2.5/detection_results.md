# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 4.1% | 3.7% |
| Cards with at least one full hit | 7.4% | 5.7% |
| Overall full-hit rate | 10 / 192 (5.2%) | 7 / 192 (3.6%) |

Card outcomes (real wins / fake wins / ties): 6 / 4 / 112. Non-tied cards: 10. One-sided exact sign-test p-value: 0.376953.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
