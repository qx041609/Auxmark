# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 29.6% | 7.1% |
| Cards with at least one full hit | 29.9% | 7.7% |
| Overall full-hit rate | 89 / 387 (23.0%) | 26 / 387 (6.7%) |

Card outcomes (real wins / fake wins / ties): 65 / 4 / 205. Non-tied cards: 69. One-sided exact sign-test p-value: 1.55737e-15.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
