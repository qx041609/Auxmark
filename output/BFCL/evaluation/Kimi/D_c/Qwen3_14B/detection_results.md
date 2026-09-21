# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 27.5% | 8.6% |
| Cards with at least one full hit | 37.9% | 10.3% |
| Overall full-hit rate | 102 / 379 (26.9%) | 26 / 379 (6.9%) |

Card outcomes (real wins / fake wins / ties): 46 / 4 / 95. Non-tied cards: 50. One-sided exact sign-test p-value: 2.23089e-10.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
