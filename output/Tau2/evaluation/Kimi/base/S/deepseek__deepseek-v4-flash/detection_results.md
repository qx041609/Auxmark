# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 7.0% | 4.9% |
| Cards with at least one full hit | 8.2% | 5.7% |
| Overall full-hit rate | 11 / 192 (5.7%) | 8 / 192 (4.2%) |

Card outcomes (real wins / fake wins / ties): 4 / 1 / 117. Non-tied cards: 5. One-sided exact sign-test p-value: 0.1875.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
