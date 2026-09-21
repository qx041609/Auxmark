# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 0.2% | 0.7% |
| Cards with at least one full hit | 0.7% | 1.4% |
| Overall full-hit rate | 1 / 379 (0.3%) | 3 / 379 (0.8%) |

Card outcomes (real wins / fake wins / ties): 1 / 2 / 142. Non-tied cards: 3. One-sided exact sign-test p-value: 0.875.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
