# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 25.9% | 3.8% |
| Cards with at least one full hit | 26.6% | 4.7% |
| Overall full-hit rate | 76 / 387 (19.6%) | 15 / 387 (3.9%) |

Card outcomes (real wins / fake wins / ties): 65 / 6 / 203. Non-tied cards: 71. One-sided exact sign-test p-value: 6.66065e-14.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
