# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 25.5% | 5.9% |
| Cards with at least one full hit | 27.4% | 7.7% |
| Overall full-hit rate | 77 / 387 (19.9%) | 22 / 387 (5.7%) |

Card outcomes (real wins / fake wins / ties): 64 / 9 / 201. Non-tied cards: 73. One-sided exact sign-test p-value: 1.18944e-11.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
