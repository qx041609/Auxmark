# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 13.4% | 4.1% |
| Cards with at least one full hit | 14.9% | 5.7% |
| Overall full-hit rate | 48 / 437 (11.0%) | 17 / 437 (3.9%) |

Card outcomes (real wins / fake wins / ties): 33 / 8 / 221. Non-tied cards: 41. One-sided exact sign-test p-value: 5.61107e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
