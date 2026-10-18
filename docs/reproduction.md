# Reproduction Guide

This project provides only the RaNN programs used by the paper's interface experiments. Run all commands from the root of this standalone project. `scripts/run_experiment.py` reads JSON configurations, imports the selected solver, and calls its `main` function. Paper configurations leave the random seed unset; pass `--seed` for a controlled diagnostic. Direct entry points retain their original repetition behavior.

## Figure Configurations

Run `python scripts/run_experiment.py configs/figure_XX.json`, replacing `XX` with the two-digit number below.

| Figure | Example | Configuration and model |
| --- | --- | --- |
| 3 | 4.1 Case 1 | `figure_03.json`: flower, m=320, N=40, beta=(1e-4,1e4) |
| 4 | 4.1 Case 2 | `figure_04.json`: circle, two sin layers, m=1280, N=100 |
| 5 | 4.2 | `figure_05.json`: sphere at z=0, m=640, N=20, beta=(1,1000) |
| 6 | 4.3 | `figure_06.json`: three interfaces, m=320, N=60, beta=(1,2,3,4) |
| 7 | 4.4 | `figure_07.json`: tanh, k=(3,1), m=1280, N=100 |
| 8 | 4.4 | `figure_08.json`: tanh, k=(3,2), m=1280, N=100 |
| 9 | 4.4 | `figure_09.json`: tanh, k=(4,1), m=1800, N=200 |
| 10 | 4.4 | `figure_10.json`: tanh, k=(4,3), m=1800, N=200 |
| 11 | 4.4 | `figure_11.json`: cos, k=(4,1), m=1800, N=200 |
| 12 | 4.4 | `figure_12.json`: cos, k=(4,3), m=1800, N=200 |
| 13-17 | 4.6 | `figures_13_17.json`: moving interface, m=640, N=10, beta=(1,1) |

The new figure exporter evaluates the existing fitted models and exact-solution functions. It saves the three fields in a common layout and archives their numerical grids. The layouts need not match the publisher's typography. Figures 1-2 are methodological illustrations, not numerical solver outputs, and are not regenerated here.

## Table Configurations

Run `python scripts/reproduce_table.py configs/table_XX.json`. The case order and labels are stored explicitly in each JSON file. Ten calls are averaged per case. `--limit 1 --repeats 1` runs only the first case once for a diagnostic; it does not produce a full table.

| Tables | Solver and varied parameters |
| --- | --- |
| 1-2 | Flower: collocation count/features; coefficient pairs and initialization ranges |
| 3-4 | Mixed flower: collocation count/features; coefficient pairs and four ranges |
| 5 | Circle: all six depth/activation choices, with m=1280 and N=40,60,80,100 |
| 7-10 | Sphere: collocation count/features, beta2=100 or 1000, penalties 10 or 1000 |
| 11 | Sphere: coefficient pairs |
| 14-15 | Mixed sphere: collocation count/features; coefficient pairs |
| 16-17 | Three interfaces: collocation count/features; coefficient quadruples |
| 18 | Three interfaces with automatic differentiation: coefficient quadruples |
| 19 | Three interfaces: first/second difference step-size grid |
| 20-21 | Square with tanh: collocation count/features; coefficient pairs |
| 22-23 | Square with cosine: collocation count/features and frequency pairs |
| 24-25 | Hyperplane: dimension/features; coefficient pairs |
| 26-27 | Moving interface: space-time count/features; coefficient pairs |

Table 5 uses the five clearly named solvers in `ex1_flower/ex1_case2/`. `circle_2d_1l.py` covers the two single-layer columns through `--outer-activation tanh` with r2=5 and `--outer-activation sin` with r2=24. The other four filenames identify their actual exterior activations. The interior model remains a single tanh layer with r1=1.2 throughout. `table_05.json` explicitly lists every column and sampling count; no source edits are needed.

The Table 18 configuration explicitly uses `numpy_lstsq_rcond=-1`, NumPy's pre-2.0 default cutoff. Current NumPy 2 defaults truncate more small singular values and can give a higher error in this ill-conditioned AD example. Direct solver calls and configurations without this setting keep the installed NumPy default. To compare either behavior, use `--numpy-lstsq-rcond -1` with the runner or remove this field from a copied configuration. No solver expression or sampling rule is changed by this compatibility setting.

## Arguments and Sampling Counts

Append original solver arguments after `--`, for example:

```bash
python scripts/run_experiment.py configs/example_sphere.json --name sphere_small -- -n1 1000 -n2 100 -n3 600 -m 160
```

| Argument | Meaning |
| --- | --- |
| `-r1`, `-r2`, ... | Uniform initialization ranges, where exposed by the selected script |
| `-hx1`, `-hx2` | First- and second-derivative finite-difference steps |
| `-n1` | Interior sample count, except the high-dimensional special case below |
| `-n2` | Sample count on each outer boundary face |
| `-n3` | Sample count on an interface; in Example 4.3, on each of three interfaces |
| `-n_int` | Gauss order per axis, or total Monte Carlo samples in Example 4.5 |
| `-m` | Number of final hidden features per scalar network |
| `-alpha1`, `-alpha2`, ... | Diffusion coefficients using each original script's numbering |
| `-p1`, `-p2`, `-p3` | Boundary, interface, and initial-condition penalties |
| `-k1`, `-k2` | Positive integer oscillation counts in square-interface solutions |
| `-dim`, `-m_d` | Dimension and sampling multiplier in Example 4.5 |
| `--outer-activation` | Exterior tanh/sin choice for the single-layer circle solver |

For 2-D single-interface cases, set `(n1,n2,n3)=(N^2,N,4N)`, giving `Nc=N^2+8N`. For the 3-D sphere, use `(N^3,N^2,6N^2)`, giving `Nc=N^3+12N^2`. For the three-interface case, use `(N^2,N,4N)` for each interface, giving `Nc=N^2+16N`. The moving case uses `(N^3,N^2,5N^2)`, giving `Nc=N^3+10N^2` after spatial boundaries and the initial plane are counted. The high-dimensional script uses its own dimension-dependent sampling formulas; keep `n1=10` and `m_d=6` for the supplied Table 24-25 configurations.

## Reproducibility and Runtime

The paper reports ten-run averages on an Intel Core i7-9700 at 3.00 GHz with 16 GB RAM. Its experiments did not fix a random seed. Compare repeated-run means, ranges, and error scales rather than expecting identical paper digits. Floating-point results can also differ across operating systems, BLAS implementations, thread counts, dependency versions, and least-squares rank cutoffs. Record the actual environment alongside results. Representative paper-sized checks are documented in [accuracy.md](accuracy.md).

The 40,000-point, 1,800-feature square runs assemble dense systems with thousands of columns. They may need several GB of memory. Automatic differentiation in Example 4.3 can be much slower than finite differences. Avoid launching all full sweeps simultaneously. The runner uses a noninteractive Matplotlib backend and closes saved figures.

Original `time.clock` calls receive a Windows-compatible performance-counter fallback on modern Python. Original solvers, LAPACK driver choices, quadrature rules, initialization order, and floating-point expressions are retained. Runtime measurements and newly exported file bytes are not part of the numerical equality claim.
