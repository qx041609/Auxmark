# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 3.1% | 1.4% |
| Cards with at least one full hit | 3.4% | 1.9% |
| Overall full-hit rate | 14 / 437 (3.2%) | 7 / 437 (1.6%) |

Card outcomes (real wins / fake wins / ties): 9 / 4 / 249. Non-tied cards: 13. One-sided exact sign-test p-value: 0.133423.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
