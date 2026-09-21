# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 60.6% | 51.3% |
| Cards with at least one full hit | 61.9% | 54.2% |
| Overall full-hit rate | 106 / 182 (58.2%) | 88 / 182 (48.4%) |

Card outcomes (real wins / fake wins / ties): 18 / 6 / 94. Non-tied cards: 24. One-sided exact sign-test p-value: 0.0113279.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
