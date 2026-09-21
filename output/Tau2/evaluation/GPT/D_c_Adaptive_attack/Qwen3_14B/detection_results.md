# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 53.1% | 17.7% |
| Cards with at least one full hit | 55.2% | 20.0% |
| Overall full-hit rate | 74 / 146 (50.7%) | 27 / 146 (18.5%) |

Card outcomes (real wins / fake wins / ties): 51 / 4 / 70. Non-tied cards: 55. One-sided exact sign-test p-value: 1.02371e-11.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
