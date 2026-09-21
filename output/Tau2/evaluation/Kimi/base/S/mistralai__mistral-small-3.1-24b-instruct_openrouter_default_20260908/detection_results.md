# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 11.5% | 9.4% |
| Cards with at least one full hit | 15.6% | 12.3% |
| Overall full-hit rate | 19 / 192 (9.9%) | 16 / 192 (8.3%) |

Card outcomes (real wins / fake wins / ties): 14 / 11 / 97. Non-tied cards: 25. One-sided exact sign-test p-value: 0.345019.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
