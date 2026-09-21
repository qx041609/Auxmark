# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 22.6% | 5.3% |
| Cards with at least one full hit | 24.1% | 5.8% |
| Overall full-hit rate | 67 / 387 (17.3%) | 18 / 387 (4.7%) |

Card outcomes (real wins / fake wins / ties): 54 / 4 / 216. Non-tied cards: 58. One-sided exact sign-test p-value: 1.58498e-12.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
