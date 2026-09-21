# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 1.2% | 0.1% |
| Cards with at least one full hit | 1.9% | 0.4% |
| Overall full-hit rate | 5 / 437 (1.1%) | 1 / 437 (0.2%) |

Card outcomes (real wins / fake wins / ties): 4 / 0 / 258. Non-tied cards: 4. One-sided exact sign-test p-value: 0.0625.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
