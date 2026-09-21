# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 0.9% | 0.9% |
| Cards with at least one full hit | 1.5% | 1.1% |
| Overall full-hit rate | 4 / 387 (1.0%) | 5 / 387 (1.3%) |

Card outcomes (real wins / fake wins / ties): 4 / 3 / 267. Non-tied cards: 7. One-sided exact sign-test p-value: 0.5.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
