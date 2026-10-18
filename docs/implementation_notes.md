# Implementation Notes

## Scope of the Refactor

The public project contains 15 solvers grouped by the paper's six examples. Circle filenames identify the actual exterior depth and activations, and the cosine solver is grouped with the square example. Duplicate experiments, unrelated network variants, symbolic scratch calculations, and unused spreadsheets are excluded. Chinese comments were translated to English. A small source-path bootstrap lets every script find `src/` from an arbitrary working directory.

The retained original syntax is recorded in `tests/fixtures/original_ast.json`. The preservation test removes the new module description/path bootstrap/compatibility import and expands shared functions before comparing syntax trees. For the single-layer circle, it additionally verifies and removes only the explicit optional sine branch, parser option, and argument forwarding. Its default tanh behavior, model initialization order, and all solver expressions remain unchanged.

## Common Function Families

| Family | Public location | Preservation choice |
| --- | --- | --- |
| First/second Cartesian differences and normal differences | `src/lrann_interface/finite_difference.py` | Only exactly matching bodies are shared; distinct slicing and return-field variants remain distinct |
| 2-D/3-D Gauss nodes and weights | `src/lrann_interface/quadrature.py` | Original Cartesian-product order and sequential multiplication are retained |
| Hidden networks and activation choices | Each experiment | Keep constructors and initialization order local to avoid changing RNG consumption |
| Uniform initialization | Each experiment | Keep script-specific global ranges and the original `apply` order |
| Boundary/interior/interface sampling | Each experiment | Keep geometry globals, point ordering, and random-number draws unchanged |
| Exact solutions, gradients, forcing terms | Each experiment | Keep coefficient conventions and expression grouping local |
| Matrix assembly and least squares | Each experiment | Keep NumPy/SciPy solver choice, default driver, shapes, and row ordering |
| Error metrics | Each experiment, collected by `runner.scalar_metrics` | Collect values already computed by the solver, without recomputing the norm |
| Configurations, logs, figures, run means | `runner.py`, `plotting.py` | New orchestration around the original solver functions |

Only helper variants imported by the retained solvers remain. Functions named `*_variant2` and `*_variant3` are intentionally separate when their original bodies differ. No algebraic simplification is applied. Models sometimes return tuples and sometimes tensors; directional-difference helpers must not be merged solely because their names look similar.

## Running and Exporting

`main_arguments` uses the selected script's argument parser and reads its main-call keyword mapping. The single-layer circle exposes `--outer-activation tanh` or `sin` to cover both published columns. Its tanh default and r2=5 are preserved; the sine paper configuration explicitly selects r2=24. The other four Table 5 architectures use their corresponding original solver bodies and initialization ranges.

Paper configurations use `seed: null`. The smoke test and exact before/after regression use fixed seeds for testing only. A user may set a diagnostic seed without changing the normal unseeded paper workflow.

The original sources call `numpy.linalg.lstsq` without a cutoff argument. NumPy 2 changed its default from machine epsilon to machine epsilon times the larger matrix dimension. The runner's optional `numpy_lstsq_rcond` setting restores the old cutoff explicitly and temporarily; absent that setting, the original call is untouched. Table 18 selects -1 for compatibility with the historical default. This option affects numerical rank and may change fitted coefficients; it is not part of the unchanged-default numerical fingerprint. Both policies are measured in the accuracy report.

Some square scripts declare `-k1` and `-k2` as floats but call `range(k)`. The configuration runner accepts only positive integral values and passes them as integers. Direct calls preserve the original parser, including its limitation when users explicitly override those flags. This avoids changing any original solver expressions.

The runner observes the solver's local variables at return using a temporary profile callback, restoring the previous callback afterward. It records existing metrics and optionally copies assembled arrays. It redirects only `plt.show` for saving figures, restores plotting state/default dtype/thread settings afterward, and never changes matrix values or fitted model weights. The callback and figure I/O can affect elapsed timing, so timing comparisons should use the original script timer and report the execution context.

Several scripts return before their commented plotting sections. `plotting.py` creates fresh field exports after the solver returns, using the captured fitted models, original exact-solution functions, the original pointwise network evaluation size, and the paper's geometry. These additional exports do not change the computed least-squares solution or error metrics.

## Publishing

Publish this directory as its own repository, for example `lrann-fdm-interface-paper-code`. The root `.gitignore` excludes local manuscripts, lab guidance documents, IDE state, caches, and generated output directories. Only the paper-related solver files and their supporting utilities/configurations belong in this release.

The code license is copied unchanged from the lab's existing BSD 3-Clause license. `CITATION.cff` cites the published article DOI. A software repository URL, release date, versioned tag, or software DOI should only be added once it exists; none is fabricated here. GitHub-only publication is the scope of this project preparation. The separate lab archive/DOI instructions describe an additional Zenodo workflow that has not been performed.
