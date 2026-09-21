# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 46.6% | 33.9% |
| Cards with at least one full hit | 47.5% | 34.7% |
| Overall full-hit rate | 82 / 182 (45.1%) | 57 / 182 (31.3%) |

Card outcomes (real wins / fake wins / ties): 24 / 7 / 87. Non-tied cards: 31. One-sided exact sign-test p-value: 0.00166345.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
