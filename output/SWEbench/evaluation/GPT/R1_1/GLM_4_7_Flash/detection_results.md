# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 31.4% | 5.1% |
| Cards with at least one full hit | 32.5% | 6.2% |
| Overall full-hit rate | 93 / 387 (24.0%) | 18 / 387 (4.7%) |

Card outcomes (real wins / fake wins / ties): 77 / 3 / 194. Non-tied cards: 80. One-sided exact sign-test p-value: 7.06421e-20.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
