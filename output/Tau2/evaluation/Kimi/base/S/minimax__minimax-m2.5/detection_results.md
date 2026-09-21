# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 13.5% | 10.2% |
| Cards with at least one full hit | 18.0% | 14.8% |
| Overall full-hit rate | 23 / 192 (12.0%) | 18 / 192 (9.4%) |

Card outcomes (real wins / fake wins / ties): 9 / 5 / 108. Non-tied cards: 14. One-sided exact sign-test p-value: 0.211975.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
