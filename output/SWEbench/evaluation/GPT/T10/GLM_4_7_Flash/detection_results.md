# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 32.2% | 6.9% |
| Cards with at least one full hit | 33.2% | 7.7% |
| Overall full-hit rate | 96 / 387 (24.8%) | 24 / 387 (6.2%) |

Card outcomes (real wins / fake wins / ties): 76 / 6 / 192. Non-tied cards: 82. One-sided exact sign-test p-value: 7.8435e-17.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
