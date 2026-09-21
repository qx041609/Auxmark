# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 42.8% | 19.1% |
| Cards with at least one full hit | 48.3% | 24.8% |
| Overall full-hit rate | 155 / 379 (40.9%) | 69 / 379 (18.2%) |

Card outcomes (real wins / fake wins / ties): 47 / 4 / 94. Non-tied cards: 51. One-sided exact sign-test p-value: 1.20815e-10.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
