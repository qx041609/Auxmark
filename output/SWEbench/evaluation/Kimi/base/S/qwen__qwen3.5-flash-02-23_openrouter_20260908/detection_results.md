# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 2.9% | 2.5% |
| Cards with at least one full hit | 4.6% | 3.8% |
| Overall full-hit rate | 15 / 437 (3.4%) | 13 / 437 (3.0%) |

Card outcomes (real wins / fake wins / ties): 6 / 4 / 252. Non-tied cards: 10. One-sided exact sign-test p-value: 0.376953.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
