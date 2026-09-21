# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 63.6% | 37.7% |
| Cards with at least one full hit | 65.3% | 39.8% |
| Overall full-hit rate | 110 / 182 (60.4%) | 63 / 182 (34.6%) |

Card outcomes (real wins / fake wins / ties): 34 / 1 / 83. Non-tied cards: 35. One-sided exact sign-test p-value: 1.04774e-09.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
