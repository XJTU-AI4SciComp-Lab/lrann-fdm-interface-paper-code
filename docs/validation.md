# Validation Record

## Original-Code Preservation

The 15 retained paper solvers were selected from the author's original source archive. Their original syntax hashes and source locations are recorded in `tests/fixtures/original_ast.json`. The preservation test expands shared helpers and removes only the import bootstrap, module description, timer compatibility import, and explicitly verified optional single-layer sine branch before comparing syntax trees.

Default formulas, model initialization order, sampling/RNG calls, matrix assembly, quadrature order, least-squares calls, and error calculations are preserved. The single-layer circle keeps its original tanh default and exposes the previously commented sine choice for the second Table 5 column. The optional historical NumPy cutoff is an explicit configuration choice, not a change to the source default.

## Numerical Fingerprints

Every retained solver passed a byte-for-byte comparison against the untouched original under the same RNG state and numerical environment. The complete public checker was rerun after curation and all 15 solvers reported `IDENTICAL`. Its reference includes 770 named local arrays/tensors/model states and individual least-squares-array fingerprints in `tests/fixtures/numerical_comparison.json`.

The comparison uses one solver call with N1=64, N2=8, N3=32, integration order 4, and m=8 where exposed. The high-dimensional case uses dimension 3, N1=4, sampling multiplier 2, and 32 Monte Carlo points. The initial default dtype is float32; original dtype changes are retained. The seed before importing each script is 2026, with one PyTorch thread. Timing values are excluded.

The reference environment is Python 3.12.7, PyTorch 2.5.1 CPU, NumPy 2.1.3, SciPy 1.14.1, Matplotlib 3.9.2, Windows x64. Small-run fingerprints are specific to that environment and its installed NumPy default cutoff. Their purpose is to detect refactoring regressions, separately from reproducing a historical paper value.

```bash
python scripts/check_numerics.py
python scripts/check_numerics.py --all
```

The all-solvers check can take a minute because two original routines evaluate 90,000-point plotting grids in Python loops. No fixed seed is imposed on normal paper configurations.

## Paper Accuracy

The paper-sized checks cover all six examples, both mixed formulations, the AD comparison, both square activations, and all six Table 5 architectures: 19 cases and 148 individual solves in total. Most selected entries use ten unseeded runs. Architecture checks use three runs per column. Individual values, means, ranges, paper references, and cutoff policies are documented in [accuracy.md](accuracy.md), with all measurements retained in small CSV files.

The AD comparison was measured with both the NumPy 2 default and the historical rcond=-1 cutoff. Two independent ten-run batches of the smaller cosine cell did not reproduce the paper's mean in this environment; this SciPy-solved case is retained alongside the matching dense cosine case. Numerical agreement is reported from the actual runs, without requiring identical random draws or selecting the best errors.

## Runtime and Artifact Tests

The 21 tests cover all retained imports, syntax preservation, finite differences, quadrature, integer oscillation arguments, the explicit cutoff policy, paper configuration paths/architectures, repeated-seed numerical equality, least-squares residuals, and saved PNG/PDF/array artifacts.

```bash
python -m pytest tests -q -p no:cacheprovider
```

Reduced figure checks cover all interface geometries, including the final-time moving interface. Generated fields were checked for finite values and PNGs for nonblank pixels. Representative multiple-interface and moving-interface exports were visually inspected. Standalone-copy checks verify that no source from the parent SO-RaNN project is required.

## Scope

Selected full-sized table cells were executed; complete table sweeps and all figure-sized experiments were not rerun. CPU time, file metadata, and figure typography are not numerical equality claims. Bitwise comparisons across different BLAS implementations, hardware, library versions, or cutoff policies are not expected. The supplied Windows/Linux CI workflow is for future repository pushes and has not been run on hosted runners here.
