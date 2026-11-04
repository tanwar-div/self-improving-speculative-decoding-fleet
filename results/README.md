# Results

Curated output from benchmark and correctness runs, committed so the figures
quoted in the top-level README can be checked against their source data.

`runs/` (git-ignored) holds raw scratch from every run; only the reports the
README actually cites are promoted here.

| file | what it is |
|---|---|
| `precision_float32.json` | greedy-equivalence precision study, fp32 |
| `precision_float16.json` | greedy-equivalence precision study, fp16, incl. the no-speculation batch-shape control |
| `benchmark.json` | full benchmark: E1a non-stationary, E1b stationary control, E2 workload-shift recovery |
| `benchmark.txt` | the same run, rendered |
| `fleet_loop.txt` | containerized fleet driven through a workload shift: retrain, canary, SLO gate, promote/rollback, hot-swap |

Reproduce with `make precision`, `make bench`, and `make fleet-up` +
`scripts/drive_traffic.py --shift`. Hardware is recorded in each JSON's
`config` block; the numbers are from an RTX 3050 Laptop (4 GB).
