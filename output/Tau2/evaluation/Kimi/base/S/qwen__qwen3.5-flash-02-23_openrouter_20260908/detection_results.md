# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 5.1% | 2.0% |
| Cards with at least one full hit | 6.6% | 2.5% |
| Overall full-hit rate | 9 / 192 (4.7%) | 3 / 192 (1.6%) |

Card outcomes (real wins / fake wins / ties): 6 / 1 / 115. Non-tied cards: 7. One-sided exact sign-test p-value: 0.0625.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
