# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 3.4% | 0.8% |
| Cards with at least one full hit | 4.2% | 0.8% |
| Overall full-hit rate | 5 / 182 (2.7%) | 1 / 182 (0.5%) |

Card outcomes (real wins / fake wins / ties): 4 / 0 / 114. Non-tied cards: 4. One-sided exact sign-test p-value: 0.0625.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
