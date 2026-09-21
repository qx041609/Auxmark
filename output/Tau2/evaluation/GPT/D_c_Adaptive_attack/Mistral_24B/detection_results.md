# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 53.7% | 26.4% |
| Cards with at least one full hit | 54.4% | 27.2% |
| Overall full-hit rate | 77 / 146 (52.7%) | 43 / 146 (29.5%) |

Card outcomes (real wins / fake wins / ties): 38 / 4 / 83. Non-tied cards: 42. One-sided exact sign-test p-value: 2.82657e-08.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
