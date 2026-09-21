# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 29.3% | 5.7% |
| Cards with at least one full hit | 30.3% | 6.9% |
| Overall full-hit rate | 87 / 387 (22.5%) | 21 / 387 (5.4%) |

Card outcomes (real wins / fake wins / ties): 68 / 4 / 202. Non-tied cards: 72. One-sided exact sign-test p-value: 2.31041e-16.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
