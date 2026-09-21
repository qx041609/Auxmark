# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 52.5% | 33.9% |
| Cards with at least one full hit | 55.1% | 35.6% |
| Overall full-hit rate | 91 / 182 (50.0%) | 59 / 182 (32.4%) |

Card outcomes (real wins / fake wins / ties): 29 / 4 / 85. Non-tied cards: 33. One-sided exact sign-test p-value: 5.4643e-06.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
