# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 33.2% | 21.3% |
| Cards with at least one full hit | 34.4% | 24.0% |
| Overall full-hit rate | 48 / 146 (32.9%) | 31 / 146 (21.2%) |

Card outcomes (real wins / fake wins / ties): 22 / 7 / 96. Non-tied cards: 29. One-sided exact sign-test p-value: 0.00406503.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
