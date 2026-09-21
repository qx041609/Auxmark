# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 36.1% | 15.6% |
| Cards with at least one full hit | 41.8% | 18.9% |
| Overall full-hit rate | 56 / 192 (29.2%) | 28 / 192 (14.6%) |

Card outcomes (real wins / fake wins / ties): 33 / 6 / 83. Non-tied cards: 39. One-sided exact sign-test p-value: 7.14963e-06.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
