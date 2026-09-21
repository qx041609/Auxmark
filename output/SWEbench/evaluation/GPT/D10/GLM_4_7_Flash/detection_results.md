# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 23.4% | 2.8% |
| Cards with at least one full hit | 24.1% | 3.6% |
| Overall full-hit rate | 68 / 387 (17.6%) | 12 / 387 (3.1%) |

Card outcomes (real wins / fake wins / ties): 60 / 4 / 210. Non-tied cards: 64. One-sided exact sign-test p-value: 3.68152e-14.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
