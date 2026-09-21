# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 29.2% | 20.9% |
| Cards with at least one full hit | 30.4% | 22.4% |
| Overall full-hit rate | 43 / 146 (29.5%) | 33 / 146 (22.6%) |

Card outcomes (real wins / fake wins / ties): 17 / 7 / 101. Non-tied cards: 24. One-sided exact sign-test p-value: 0.0319573.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
