# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 30.5% | 10.0% |
| Cards with at least one full hit | 32.1% | 11.7% |
| Overall full-hit rate | 94 / 387 (24.3%) | 37 / 387 (9.6%) |

Card outcomes (real wins / fake wins / ties): 66 / 11 / 197. Non-tied cards: 77. One-sided exact sign-test p-value: 5.26994e-11.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
