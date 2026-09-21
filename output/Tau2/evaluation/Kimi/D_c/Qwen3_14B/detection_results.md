# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 37.7% | 19.7% |
| Cards with at least one full hit | 41.8% | 22.1% |
| Overall full-hit rate | 60 / 192 (31.2%) | 36 / 192 (18.8%) |

Card outcomes (real wins / fake wins / ties): 33 / 9 / 80. Non-tied cards: 42. One-sided exact sign-test p-value: 0.00013577.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
