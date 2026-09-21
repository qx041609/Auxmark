# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 25.1% | 8.5% |
| Cards with at least one full hit | 26.6% | 9.9% |
| Overall full-hit rate | 77 / 387 (19.9%) | 30 / 387 (7.8%) |

Card outcomes (real wins / fake wins / ties): 54 / 8 / 212. Non-tied cards: 62. One-sided exact sign-test p-value: 8.54663e-10.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
