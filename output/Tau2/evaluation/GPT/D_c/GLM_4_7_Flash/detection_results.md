# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 58.4% | 28.8% |
| Cards with at least one full hit | 60.0% | 30.4% |
| Overall full-hit rate | 76 / 146 (52.1%) | 42 / 146 (28.8%) |

Card outcomes (real wins / fake wins / ties): 42 / 4 / 79. Non-tied cards: 46. One-sided exact sign-test p-value: 2.5501e-09.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
