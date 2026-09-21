# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 21.4% | 3.8% |
| Cards with at least one full hit | 22.3% | 4.7% |
| Overall full-hit rate | 63 / 387 (16.3%) | 15 / 387 (3.9%) |

Card outcomes (real wins / fake wins / ties): 53 / 6 / 215. Non-tied cards: 59. One-sided exact sign-test p-value: 8.76959e-11.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
