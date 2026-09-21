# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 20.4% | 2.9% |
| Cards with at least one full hit | 21.9% | 4.0% |
| Overall full-hit rate | 66 / 387 (17.1%) | 11 / 387 (2.8%) |

Card outcomes (real wins / fake wins / ties): 54 / 2 / 218. Non-tied cards: 56. One-sided exact sign-test p-value: 2.21628e-14.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
