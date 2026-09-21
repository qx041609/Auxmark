# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 0.5% | 0.7% |
| Cards with at least one full hit | 1.4% | 0.7% |
| Overall full-hit rate | 2 / 379 (0.5%) | 2 / 379 (0.5%) |

Card outcomes (real wins / fake wins / ties): 2 / 1 / 142. Non-tied cards: 3. One-sided exact sign-test p-value: 0.5.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
