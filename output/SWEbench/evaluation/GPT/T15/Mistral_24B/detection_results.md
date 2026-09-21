# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 32.2% | 11.1% |
| Cards with at least one full hit | 34.3% | 12.8% |
| Overall full-hit rate | 101 / 387 (26.1%) | 40 / 387 (10.3%) |

Card outcomes (real wins / fake wins / ties): 71 / 12 / 191. Non-tied cards: 83. One-sided exact sign-test p-value: 1.19732e-11.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
