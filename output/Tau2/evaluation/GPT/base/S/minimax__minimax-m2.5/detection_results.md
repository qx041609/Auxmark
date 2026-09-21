# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 15.6% | 13.2% |
| Cards with at least one full hit | 16.8% | 13.6% |
| Overall full-hit rate | 21 / 146 (14.4%) | 19 / 146 (13.0%) |

Card outcomes (real wins / fake wins / ties): 6 / 3 / 116. Non-tied cards: 9. One-sided exact sign-test p-value: 0.253906.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
