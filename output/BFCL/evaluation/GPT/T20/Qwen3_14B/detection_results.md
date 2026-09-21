# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 31.8% | 21.6% |
| Cards with at least one full hit | 33.9% | 22.9% |
| Overall full-hit rate | 49 / 182 (26.9%) | 31 / 182 (17.0%) |

Card outcomes (real wins / fake wins / ties): 16 / 1 / 101. Non-tied cards: 17. One-sided exact sign-test p-value: 0.000137329.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
