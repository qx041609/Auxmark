# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 3.5% | 1.9% |
| Cards with at least one full hit | 4.4% | 2.9% |
| Overall full-hit rate | 15 / 387 (3.9%) | 10 / 387 (2.6%) |

Card outcomes (real wins / fake wins / ties): 10 / 6 / 258. Non-tied cards: 16. One-sided exact sign-test p-value: 0.227249.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
