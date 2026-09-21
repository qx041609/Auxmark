# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 14.6% | 5.3% |
| Cards with at least one full hit | 15.7% | 6.2% |
| Overall full-hit rate | 45 / 387 (11.6%) | 20 / 387 (5.2%) |

Card outcomes (real wins / fake wins / ties): 31 / 6 / 237. Non-tied cards: 37. One-sided exact sign-test p-value: 2.06288e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
