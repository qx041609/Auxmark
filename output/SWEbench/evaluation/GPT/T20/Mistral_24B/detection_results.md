# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 28.9% | 10.0% |
| Cards with at least one full hit | 30.3% | 10.9% |
| Overall full-hit rate | 90 / 387 (23.3%) | 36 / 387 (9.3%) |

Card outcomes (real wins / fake wins / ties): 58 / 5 / 211. Non-tied cards: 63. One-sided exact sign-test p-value: 8.31175e-13.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
