# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 19.4% | 3.5% |
| Cards with at least one full hit | 20.4% | 4.0% |
| Overall full-hit rate | 59 / 387 (15.2%) | 13 / 387 (3.4%) |

Card outcomes (real wins / fake wins / ties): 47 / 2 / 225. Non-tied cards: 49. One-sided exact sign-test p-value: 2.17781e-12.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
