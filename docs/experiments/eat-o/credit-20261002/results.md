# EAT-O exact discrete credit results

All 11 checkpoint evaluations and the paired comparison passed independent checks.
Same 20,000-example recipe; five paired seeds. Validation was reused. No matched-work qualification.

| Signal | Mean accuracy | SD, pp | Mean loss | Mean counted work |
|---|---:|---:|---:|---:|
| backprop | 73.300% | 4.896 | 41007.366 | 3,204,736,267.4 |
| probe | 66.480% | 1.595 | 46447.410 | 3,218,155,488.0 |

Probe minus backprop loss: 5440.044; paired 95% t interval [2180.129, 8699.959].
The raw-cache and reference probe seed-1 models are identical. Intervals are exploratory, unadjusted for repeated experiments.

## Every run

| Route | Seed | Accuracy | Mean loss | Work | Transitions | Reversals |
|---|---:|---:|---:|---:|---:|---:|
| backprop | 1 | 68.800% | 41047.507 | 3,203,190,434 | 158,654 | 90,450 |
| backprop | 2 | 74.975% | 38972.272 | 3,204,926,422 | 163,198 | 93,200 |
| backprop | 3 | 67.575% | 43378.439 | 3,205,656,365 | 161,742 | 92,024 |
| backprop | 4 | 78.900% | 38616.311 | 3,205,539,815 | 178,755 | 102,609 |
| backprop | 5 | 76.250% | 43022.302 | 3,204,368,301 | 158,132 | 89,819 |
| probe | 1 | 65.875% | 45707.821 | 3,218,231,688 | 90,248 | 41,577 |
| probe | 2 | 67.075% | 46920.287 | 3,217,787,402 | 93,171 | 43,941 |
| probe | 3 | 65.500% | 46312.090 | 3,217,906,658 | 95,086 | 44,419 |
| probe | 4 | 64.975% | 47099.586 | 3,219,449,162 | 95,391 | 43,766 |
| probe | 5 | 68.975% | 46197.267 | 3,217,402,530 | 90,944 | 41,749 |
| probe_reference | 1 | 65.875% | 45707.821 | 233,401,937,952 | 90,248 | 41,577 |

[Protocol](protocol.json), [CSV](runs.csv), [events](events.jsonl), [control](control.json), [independent checks](independent-analysis.json).
Each run also has a complete report, command and verification receipt. Checkpoints remain local under the protocol artifact path.
