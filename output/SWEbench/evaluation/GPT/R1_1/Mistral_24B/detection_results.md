# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 31.4% | 9.1% |
| Cards with at least one full hit | 33.2% | 10.9% |
| Overall full-hit rate | 99 / 387 (25.6%) | 33 / 387 (8.5%) |

Card outcomes (real wins / fake wins / ties): 68 / 6 / 200. Non-tied cards: 74. One-sided exact sign-test p-value: 1.07244e-14.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
