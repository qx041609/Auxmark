# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 24.9% | 5.2% |
| Cards with at least one full hit | 27.0% | 6.9% |
| Overall full-hit rate | 76 / 387 (19.6%) | 23 / 387 (5.9%) |

Card outcomes (real wins / fake wins / ties): 63 / 8 / 203. Non-tied cards: 71. One-sided exact sign-test p-value: 5.13568e-12.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
