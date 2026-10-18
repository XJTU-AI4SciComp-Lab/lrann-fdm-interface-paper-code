# Minimal Example

Run `python examples/run_smoke.py` from the project root. The script resolves its configuration and source paths relative to its own location, so it also works when invoked by absolute path from another working directory.

`configs/smoke.json` selects the square-interface solver with 64 interior points, 8 points per outer edge, 32 interface points, 16 hidden features, seed 2026, and one CPU thread. It evaluates the original quadrature error and exports a 24-by-24 field preview. Reduced parameters demonstrate the workflow, not paper-level accuracy.

Outputs are written to `results/smoke/` and `figures/smoke/`. Rerunning the same configuration overwrites that run directory's files. Choose another `name` in a configuration or pass `--name` to `scripts/run_experiment.py` to retain a separate run.
