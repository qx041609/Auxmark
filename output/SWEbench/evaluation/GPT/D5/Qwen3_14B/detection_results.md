# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 25.7% | 3.0% |
| Cards with at least one full hit | 27.4% | 4.4% |
| Overall full-hit rate | 79 / 387 (20.4%) | 13 / 387 (3.4%) |

Card outcomes (real wins / fake wins / ties): 68 / 3 / 203. Non-tied cards: 71. One-sided exact sign-test p-value: 2.5289e-17.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
