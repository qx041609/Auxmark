# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 49.6% | 33.9% |
| Cards with at least one full hit | 50.8% | 34.7% |
| Overall full-hit rate | 89 / 182 (48.9%) | 60 / 182 (33.0%) |

Card outcomes (real wins / fake wins / ties): 21 / 3 / 94. Non-tied cards: 24. One-sided exact sign-test p-value: 0.000138581.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
