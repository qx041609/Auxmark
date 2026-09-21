# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 26.0% | 5.5% |
| Cards with at least one full hit | 27.7% | 6.9% |
| Overall full-hit rate | 79 / 387 (20.4%) | 20 / 387 (5.2%) |

Card outcomes (real wins / fake wins / ties): 64 / 6 / 204. Non-tied cards: 70. One-sided exact sign-test p-value: 1.22136e-13.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
