# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 32.8% | 4.1% |
| Cards with at least one full hit | 34.3% | 4.7% |
| Overall full-hit rate | 96 / 387 (24.8%) | 15 / 387 (3.9%) |

Card outcomes (real wins / fake wins / ties): 86 / 4 / 184. Non-tied cards: 90. One-sided exact sign-test p-value: 2.16227e-21.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
