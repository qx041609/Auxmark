# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 35.2% | 11.1% |
| Cards with at least one full hit | 37.2% | 12.8% |
| Overall full-hit rate | 112 / 387 (28.9%) | 40 / 387 (10.3%) |

Card outcomes (real wins / fake wins / ties): 74 / 5 / 195. Non-tied cards: 79. One-sided exact sign-test p-value: 3.99069e-17.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
