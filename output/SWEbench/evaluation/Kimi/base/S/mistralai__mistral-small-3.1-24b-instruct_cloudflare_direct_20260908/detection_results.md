# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 3.7% | 2.1% |
| Cards with at least one full hit | 5.0% | 3.1% |
| Overall full-hit rate | 20 / 437 (4.6%) | 11 / 437 (2.5%) |

Card outcomes (real wins / fake wins / ties): 10 / 4 / 248. Non-tied cards: 14. One-sided exact sign-test p-value: 0.0897827.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
