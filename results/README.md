# Results

The configuration runner writes one directory per configuration `name`:

```text
results/<name>/
  config_used.json   Requested seed, repeats, arguments and figure settings
  environment.json  Python, platform and numerical-library versions
  log.txt           Original solver console output
  metrics.csv       Error metrics and timing for each solver call
  summary.json      Arithmetic means across the solver calls
  run_01.npz        Optional arrays, enabled with --save-arrays
```

Array archives contain the available `matrix_a`, `matrix_b`, `wu`, `output_u`, and `u_exact` values. These are copied after computation and do not modify solver arrays. Large archives are optional and ignored by Git.

`err_L2` is the absolute quadrature/Monte Carlo error, `u_L2` is the exact-solution norm, and `err_rL2` is their ratio. Mixed scripts also report flux errors. Use absolute `err_L2` (and absolute flux error) for both sphere formulations when comparing with the paper; use relative error for the remaining examples. `ttt` is the original timer. `wall_seconds` includes evaluation and export work and is not interchangeable with the paper's CPU-time columns.

Table sweeps additionally write `results/table_XX/table.csv`. Every case has its own configuration, per-run metrics, and log. Generated output directories are ignored by Git. Preserve a run under a distinct `name` before changing its configuration.

`paper_accuracy_reference.csv` and `paper_accuracy_runs.csv` contain the measured paper-sized reference checks, including all random runs rather than selected best results. `scripts/check_paper_accuracy.py` writes a fresh `paper_accuracy_latest.csv` and normal per-case run directories. See [accuracy.md](../docs/accuracy.md) for the matching paper entries and measured ranges.
