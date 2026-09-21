# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 47.0% | 37.3% |
| Cards with at least one full hit | 47.5% | 39.0% |
| Overall full-hit rate | 83 / 182 (45.6%) | 63 / 182 (34.6%) |

Card outcomes (real wins / fake wins / ties): 17 / 4 / 97. Non-tied cards: 21. One-sided exact sign-test p-value: 0.00359869.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
