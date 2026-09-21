# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 37.1% | 23.9% |
| Cards with at least one full hit | 40.0% | 25.6% |
| Overall full-hit rate | 55 / 146 (37.7%) | 37 / 146 (25.3%) |

Card outcomes (real wins / fake wins / ties): 24 / 8 / 93. Non-tied cards: 32. One-sided exact sign-test p-value: 0.00350018.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
