# Configurations

Configurations use JSON and require no YAML parser. `smoke.json` is the small installation check with a fixed seed. `example_*.json` runs a selected paper solver once with its default arguments. Paper examples, figures, and tables use `seed: null`. `figure_*.json` selects caption settings and exports paper fields. `table_*.json` contains a base solver configuration plus the explicit parameter cases. `accuracy_checks.json` selects paper-sized checks with the published reference values and correct error metric.

Runner fields are `name`, `script`, `seed`, `repeats`, `arguments`, optional `threads`, optional `save_arrays`, optional `numpy_lstsq_rcond`, and optional `figure`. Paths in `script` are relative to the project root. `arguments` is a list of original CLI flags and values, all strings. Extra command-line arguments override earlier occurrences of the same flag. Choose a distinct `name` to keep a separate result directory.

Example:

```bash
python scripts/run_experiment.py configs/example_square.json --seed 42 --name square_seed42 -- -n1 256 -n2 16 -n3 64 -m 40
python scripts/reproduce_table.py configs/table_20.json --limit 1 --repeats 1
```

The first command sets a diagnostic seed and sample count without editing solver files. The second is a one-case diagnostic rather than the complete ten-run paper table. Full sweeps can be expensive. `table_05.json` covers all six paper architectures. `table_18.json` explicitly selects NumPy's historical cutoff with `numpy_lstsq_rcond=-1`; other tables retain the installed solver default. See [reproduction.md](../docs/reproduction.md) for the complete mapping.
