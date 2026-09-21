# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 20.5% | 10.8% |
| Cards with at least one full hit | 23.7% | 13.0% |
| Overall full-hit rate | 80 / 437 (18.3%) | 42 / 437 (9.6%) |

Card outcomes (real wins / fake wins / ties): 34 / 5 / 223. Non-tied cards: 39. One-sided exact sign-test p-value: 1.21495e-06.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
