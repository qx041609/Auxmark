# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 1.5% | 0.4% |
| Cards with at least one full hit | 1.8% | 0.4% |
| Overall full-hit rate | 5 / 387 (1.3%) | 2 / 387 (0.5%) |

Card outcomes (real wins / fake wins / ties): 4 / 1 / 269. Non-tied cards: 5. One-sided exact sign-test p-value: 0.1875.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
