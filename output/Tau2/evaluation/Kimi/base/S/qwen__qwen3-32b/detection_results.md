# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 6.0% | 2.9% |
| Cards with at least one full hit | 8.2% | 3.3% |
| Overall full-hit rate | 10 / 192 (5.2%) | 5 / 192 (2.6%) |

Card outcomes (real wins / fake wins / ties): 6 / 1 / 115. Non-tied cards: 7. One-sided exact sign-test p-value: 0.0625.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
