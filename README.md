# LRaNN-FDM Interface Paper Code

This repository contains the RaNN code for six interface-problem examples in the paper below.

## Paper Information

- Title: Local randomized neural networks with finite difference methods for interface problems
- Authors: Yunlong Li and Fei Wang
- Journal: Journal of Computational Physics 529 (2025), 113847
- DOI: [10.1016/j.jcp.2025.113847](https://doi.org/10.1016/j.jcp.2025.113847)
- Corresponding author: Fei Wang, `feiwang.xjtu@xjtu.edu.cn`
- Code maintainer: Yunlong Li

## Method Summary

LRaNN-FDM approximates the solution independently in each subdomain using randomized neural networks with fixed hidden weights. Finite differences approximate derivatives, and linear least squares determines the output weights. The experiments include mixed formulations, an automatic-differentiation comparison, depth/activation comparisons, high-dimensional problems, and a moving interface.

## Repository Structure

```text
.
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- environment.yml
|-- src/lrann_interface/
|-- experiments/
|   |-- ex1_flower/
|   |-- ex2_sphere/
|   |-- ex3_three_interface/
|   |-- ex4_square/
|   |-- ex5_high_dimension/
|   `-- ex6_moving_interface/
|-- configs/
|-- scripts/
|-- examples/
|-- data/
|-- results/
|-- figures/
|-- tests/
`-- docs/
```

`src/` contains shared finite differences, quadrature, compatibility helpers, and result export. `experiments/` contains only the 15 paper-related solvers. `configs/` records figure and table parameters; `scripts/` provides their common command-line runners. Original solver filenames and numerical expressions are retained where possible.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate lrann-fdm-interface
```

Using pip in a Python 3.12 environment:

```bash
python -m pip install -r requirements.txt
```

The pinned dependencies match the local comparison environment: Python 3.12.7, PyTorch 2.5.1 CPU, NumPy 2.1.3, SciPy 1.14.1, and Matplotlib 3.9.2. Run the commands below from this repository's root directory; no parent project is needed.

## Quick Start

Run a small end-to-end example:

```bash
python examples/run_smoke.py
```

This checks assembly, solving, error evaluation, and figure export with reduced parameters. Outputs are saved to `results/smoke/` and `figures/smoke/`; they are not paper-accuracy results.

Run the tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

## Reproduce Results

| Paper example | Directory | RaNN results |
| --- | --- | --- |
| 4.1: flower and circle | `experiments/ex1_flower/` | Figures 3-4; Tables 1-5 |
| 4.2: sphere | `experiments/ex2_sphere/` | Figure 5; Tables 7-11, 14-15 |
| 4.3: three interfaces | `experiments/ex3_three_interface/` | Figure 6; Tables 16-19 |
| 4.4: square | `experiments/ex4_square/` | Figures 7-12; Tables 20-23 |
| 4.5: high-dimensional hyperplane | `experiments/ex5_high_dimension/` | Tables 24-25 |
| 4.6: moving interface | `experiments/ex6_moving_interface/` | Figures 13-17; Tables 26-27 |

Representative entry points for the six examples:

```bash
python scripts/run_experiment.py configs/figure_03.json
python scripts/run_experiment.py configs/figure_05.json
python scripts/run_experiment.py configs/figure_06.json
python scripts/run_experiment.py configs/figure_08.json
python scripts/reproduce_table.py configs/table_24.json
python scripts/run_experiment.py configs/figures_13_17.json
```

Figure configurations use the corresponding caption parameters. Table configurations average ten solver calls per case. Generated metrics, configurations, and logs go to `results/<name>/`; PNG/PDF figures and numerical grids go to `figures/<name>/`. Plotting is noninteractive, so a display server is not required.

See the [reproduction guide](docs/reproduction.md) for all figure/table commands and the [solver inventory](docs/script_inventory.md) for the 15-file mapping and shared functions. Original scripts can also be run directly; their original defaults and repetition behavior are retained. Use the configuration runner for paper parameter selection and automatic export.

Paper configurations leave the random seed unset. Random realizations and numerical-library versions can affect accuracy; identical digits across independent runs are not expected. [Measured accuracy checks](docs/accuracy.md) include all 148 recorded solves and the small cosine-case discrepancy. [Validation notes](docs/validation.md) describe the separate refactoring-preservation checks.

## Data

No external dataset is required. Each experiment defines its geometry, coefficients, manufactured solution, and sampling rules. Small reference accuracy CSV files are included in `results/`; generated dense systems, logs, and figure directories are ignored by Git.

## Hardware Notes

The included programs run on CPU; a GPU is not required. The paper reports an Intel Core i7-9700 at 3.00 GHz with 16 GB RAM. Large square-interface configurations assemble dense least-squares systems and can require several GB of memory. Full table sweeps and automatic differentiation can take substantially longer than the smoke example; run large jobs separately.

## Citation

If you use this code, please cite the corresponding paper:

```bibtex
@article{LiWang2025LRaNNFDM,
  title = {Local randomized neural networks with finite difference methods for interface problems},
  author = {Li, Yunlong and Wang, Fei},
  journal = {Journal of Computational Physics},
  volume = {529},
  pages = {113847},
  year = {2025},
  doi = {10.1016/j.jcp.2025.113847}
}
```

## License

This project is released under the BSD 3-Clause License. See [LICENSE](LICENSE).
