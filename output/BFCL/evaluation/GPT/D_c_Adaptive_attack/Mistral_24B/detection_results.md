# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 54.7% | 46.6% |
| Cards with at least one full hit | 55.1% | 48.3% |
| Overall full-hit rate | 97 / 182 (53.3%) | 81 / 182 (44.5%) |

Card outcomes (real wins / fake wins / ties): 13 / 1 / 104. Non-tied cards: 14. One-sided exact sign-test p-value: 0.000915527.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
