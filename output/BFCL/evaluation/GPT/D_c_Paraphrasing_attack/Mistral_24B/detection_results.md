# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 44.9% | 27.5% |
| Cards with at least one full hit | 46.6% | 30.5% |
| Overall full-hit rate | 81 / 182 (44.5%) | 51 / 182 (28.0%) |

Card outcomes (real wins / fake wins / ties): 27 / 5 / 86. Non-tied cards: 32. One-sided exact sign-test p-value: 5.65371e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
