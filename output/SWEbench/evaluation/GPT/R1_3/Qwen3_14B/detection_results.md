# Paired probe detection results

| Metric | Real probes | Fake probes |
|---|---:|---:|
| Mean full-hit rate per card | 24.3% | 3.5% |
| Cards with at least one full hit | 25.2% | 4.0% |
| Overall full-hit rate | 72 / 387 (18.6%) | 12 / 387 (3.1%) |

Card outcomes (real wins / fake wins / ties): 62 / 3 / 209. Non-tied cards: 65. One-sided exact sign-test p-value: 1.24212e-15.

The p-value uses the sign of (real full hits minus fake full hits) within each release card. Multiple probes from one card are not treated as independent samples.
