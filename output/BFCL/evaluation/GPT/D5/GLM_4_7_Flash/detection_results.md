# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 64.0% | 51.3% |
| Cards with at least one full hit | 64.4% | 52.5% |
| Overall full-hit rate | 111 / 182 (61.0%) | 88 / 182 (48.4%) |

Card outcomes (real wins / fake wins / ties): 21 / 5 / 92. Non-tied cards: 26. One-sided exact sign-test p-value: 0.00124696.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
