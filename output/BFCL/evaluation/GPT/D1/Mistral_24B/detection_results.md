# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 57.2% | 37.7% |
| Cards with at least one full hit | 57.6% | 39.0% |
| Overall full-hit rate | 98 / 182 (53.8%) | 65 / 182 (35.7%) |

Card outcomes (real wins / fake wins / ties): 27 / 5 / 86. Non-tied cards: 32. One-sided exact sign-test p-value: 5.65371e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
