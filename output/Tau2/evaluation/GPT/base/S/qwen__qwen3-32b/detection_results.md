# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 10.5% | 11.7% |
| Cards with at least one full hit | 12.0% | 12.8% |
| Overall full-hit rate | 17 / 146 (11.6%) | 18 / 146 (12.3%) |

Card outcomes (real wins / fake wins / ties): 2 / 3 / 120. Non-tied cards: 5. One-sided exact sign-test p-value: 0.8125.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
