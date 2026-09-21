# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 46.6% | 29.2% |
| Cards with at least one full hit | 48.3% | 32.2% |
| Overall full-hit rate | 76 / 182 (41.8%) | 48 / 182 (26.4%) |

Card outcomes (real wins / fake wins / ties): 28 / 5 / 85. Non-tied cards: 33. One-sided exact sign-test p-value: 3.30938e-05.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
