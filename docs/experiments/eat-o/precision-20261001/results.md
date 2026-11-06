# EAT-O precision ablation results

All 25 runs completed and passed independent checkpoint evaluation and consistency checks.
Accuracy is mean ± sample standard deviation over five seeds; work is counted contraction terms.
This is exploratory evaluation on reused validation data, with no matched-work qualification.

| Policy | Accuracy | Mean loss | Mean work | Mean final state bytes | Loss difference vs adaptive24, 95% CI |
|---|---:|---:|---:|---:|---:|
| adaptive12 | 73.300% ± 4.896 | 41007.366 | 20,212,900,218.2 | 1,031,008.0 | 0.000 [0.000, 0.000] |
| adaptive24 | 73.300% ± 4.896 | 41007.366 | 20,212,900,218.2 | 1,031,008.0 | 0.000 [0.000, 0.000] |
| fixed8 | 74.650% ± 4.406 | 39823.493 | 20,533,836,351.2 | 1,029,344.0 | -1183.873 [-3590.779, 1223.033] |
| fixed24 | 73.300% ± 4.896 | 41007.366 | 20,212,900,218.2 | 2,350,560.0 | 0.000 [0.000, 0.000] |
| integer32 | 73.300% ± 4.896 | 41007.366 | 20,212,900,218.2 | 1,994,848.0 | 0.000 [0.000, 0.000] |

## Every run

| Policy | Seed | Accuracy | Mean loss | Work | State bytes | Growth | Rescalings |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive12 | 1 | 68.800% | 41047.507 | 19,646,963,470 | 1,031,840 | 815 | 0 |
| adaptive12 | 2 | 74.975% | 38972.272 | 20,119,937,814 | 1,030,176 | 780 | 0 |
| adaptive12 | 3 | 67.575% | 43378.439 | 19,969,699,253 | 1,031,008 | 783 | 0 |
| adaptive12 | 4 | 78.900% | 38616.311 | 21,733,992,005 | 1,031,008 | 793 | 0 |
| adaptive12 | 5 | 76.250% | 43022.302 | 19,593,908,549 | 1,031,008 | 793 | 0 |
| adaptive24 | 1 | 68.800% | 41047.507 | 19,646,963,470 | 1,031,840 | 815 | 0 |
| adaptive24 | 2 | 74.975% | 38972.272 | 20,119,937,814 | 1,030,176 | 780 | 0 |
| adaptive24 | 3 | 67.575% | 43378.439 | 19,969,699,253 | 1,031,008 | 783 | 0 |
| adaptive24 | 4 | 78.900% | 38616.311 | 21,733,992,005 | 1,031,008 | 793 | 0 |
| adaptive24 | 5 | 76.250% | 43022.302 | 19,593,908,549 | 1,031,008 | 793 | 0 |
| fixed8 | 1 | 71.300% | 39393.666 | 19,574,319,508 | 1,029,344 | 0 | 19 |
| fixed8 | 2 | 71.700% | 39749.550 | 19,438,297,418 | 1,029,344 | 0 | 20 |
| fixed8 | 3 | 71.550% | 41646.896 | 20,545,063,815 | 1,029,344 | 0 | 21 |
| fixed8 | 4 | 77.950% | 39221.031 | 22,659,133,057 | 1,029,344 | 0 | 21 |
| fixed8 | 5 | 80.750% | 39106.325 | 20,452,367,958 | 1,029,344 | 0 | 23 |
| fixed24 | 1 | 68.800% | 41047.507 | 19,646,963,470 | 2,350,560 | 0 | 0 |
| fixed24 | 2 | 74.975% | 38972.272 | 20,119,937,814 | 2,350,560 | 0 | 0 |
| fixed24 | 3 | 67.575% | 43378.439 | 19,969,699,253 | 2,350,560 | 0 | 0 |
| fixed24 | 4 | 78.900% | 38616.311 | 21,733,992,005 | 2,350,560 | 0 | 0 |
| fixed24 | 5 | 76.250% | 43022.302 | 19,593,908,549 | 2,350,560 | 0 | 0 |
| integer32 | 1 | 68.800% | 41047.507 | 19,646,963,470 | 1,994,848 | 0 | 0 |
| integer32 | 2 | 74.975% | 38972.272 | 20,119,937,814 | 1,994,848 | 0 | 0 |
| integer32 | 3 | 67.575% | 43378.439 | 19,969,699,253 | 1,994,848 | 0 | 0 |
| integer32 | 4 | 78.900% | 38616.311 | 21,733,992,005 | 1,994,848 | 0 | 0 |
| integer32 | 5 | 76.250% | 43022.302 | 19,593,908,549 | 1,994,848 | 0 | 0 |

## Evidence

- [Frozen protocol](protocol.json), [per-run CSV](runs.csv), [event log](events.jsonl).
- [Campaign summary](summary.json), [independent analysis](independent-analysis.json).
- Each run has a command, complete JSON report, and verification receipt in this directory.
- Confidence intervals are paired t intervals with four degrees of freedom, without multiplicity adjustment.
- Fixed modes also differ from adaptive in exponent refinement; this does not isolate mantissa width alone.
- Actual state allocation differs despite the common 24-byte cap. Replay is disabled in every run.
- Large checkpoints remain local under the artifact directory in the protocol; compact records here are durable.
