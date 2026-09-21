# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 50.4% | 31.4% |
| Cards with at least one full hit | 52.5% | 33.1% |
| Overall full-hit rate | 83 / 182 (45.6%) | 53 / 182 (29.1%) |

Card outcomes (real wins / fake wins / ties): 31 / 6 / 81. Non-tied cards: 37. One-sided exact sign-test p-value: 2.06288e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
