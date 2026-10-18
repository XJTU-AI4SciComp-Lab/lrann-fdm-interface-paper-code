# Paper Accuracy Checks

These checks use the paper's sampling counts, feature counts, coefficients, activation functions, and error definitions. No fixed seed was set and no runs were discarded or replaced based on their error. Standard checks use ten consecutive random solves; the six Table 5 architecture checks use three solves each. These are selected paper entries, not complete table sweeps.

The environment is Python 3.12.7, PyTorch 2.5.1 CPU, NumPy 2.1.3, SciPy 1.14.1, and Matplotlib 3.9.2 on Windows x64. PyTorch and BLAS use one thread. The default dtype begins as float32 for each case and then follows the original script, including float64 in subsequent runs. Time spent in legacy plotting is not used to judge numerical agreement.

## Measured Results

`err_L2` is the absolute error (both sphere formulations); `err_rL2` is relative error. The ratio is measured mean divided by the paper value, not a pass/fail threshold.

| Case | Paper entry | Metric | Runs | Paper | Mean | Min-max | Mean/paper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `circle_1l_sin` | Table 5: Nc=1920, m=1280 | `err_rL2` | 3 | 7.320e-02 | 1.677e-01 | 1.68e-02 - 4.27e-01 | 2.29 |
| `circle_1l_tanh` | Table 5: Nc=1920, m=1280 | `err_rL2` | 3 | 1.480e-01 | 8.521e-02 | 7.31e-02 - 9.49e-02 | 0.58 |
| `circle_2l_sin_sin` | Table 5: Nc=1920, m=1280 | `err_rL2` | 3 | 1.710e-02 | 6.380e-02 | 2.17e-02 - 1.42e-01 | 3.73 |
| `circle_2l_tanh_sin` | Table 5: Nc=1920, m=1280 | `err_rL2` | 3 | 5.260e-02 | 1.235e-02 | 6.07e-03 - 2.01e-02 | 0.23 |
| `circle_3l_sin_sin_sin` | Table 5: Nc=1920, m=1280 | `err_rL2` | 3 | 2.050e-01 | 4.202e-02 | 1.46e-02 - 8.83e-02 | 0.20 |
| `circle_3l_tanh_tanh_sin` | Table 5: Nc=1920, m=1280 | `err_rL2` | 3 | 4.420e-02 | 7.316e-02 | 2.91e-02 - 1.17e-01 | 1.66 |
| `cosine` | Table 22: Nc=1920, m=640 | `err_rL2` | 10 | 1.540e-04 | 1.853e-03 | 5.42e-04 - 4.61e-03 | 12.03 |
| `cosine_dense` | Table 22: Nc=10800, m=1280 | `err_rL2` | 10 | 5.130e-06 | 5.023e-06 | 3.82e-06 - 7.43e-06 | 0.98 |
| `cosine_repeat_batch2` | Table 22: Nc=1920, m=640 | `err_rL2` | 10 | 1.540e-04 | 1.057e-03 | 1.54e-04 - 2.65e-03 | 6.86 |
| `flower` | Table 1: Nc=1920, m=320 | `err_rL2` | 10 | 9.780e-08 | 8.793e-08 | 5.05e-08 - 1.17e-07 | 0.90 |
| `high_dimension` | Table 24: d=5, m=225 | `err_rL2` | 10 | 5.170e-07 | 4.795e-07 | 3.41e-07 - 6.00e-07 | 0.93 |
| `mixed_flower` | Table 3: Nc=1920, m=160 | `err_rL2` | 10 | 1.140e-07 | 3.885e-07 | 1.93e-07 - 6.15e-07 | 3.41 |
| `mixed_sphere` | Table 14: Nc=2200, m=160 | `err_L2` | 10 | 3.640e-03 | 4.832e-03 | 3.16e-03 - 8.18e-03 | 1.33 |
| `moving` | Table 26: Nc=2000, m=640 | `err_rL2` | 10 | 5.960e-06 | 5.460e-06 | 1.63e-06 - 1.04e-05 | 0.92 |
| `sphere` | Table 7: Nc=2200, m=160 | `err_L2` | 10 | 4.010e-02 | 3.872e-02 | 2.48e-02 - 7.14e-02 | 0.97 |
| `square` | Table 20: Nc=10800, m=1280 | `err_rL2` | 10 | 3.700e-05 | 5.206e-05 | 3.20e-05 - 9.22e-05 | 1.41 |
| `three` | Table 17: beta=(1,2,3,4) | `err_rL2` | 10 | 4.350e-09 | 3.005e-09 | 2.14e-09 - 3.73e-09 | 0.69 |
| `three_ad` | Table 18: beta=(1,2,3,4) | `err_rL2` | 10 | 2.730e-10 | 2.719e-09 | 1.04e-09 - 5.00e-09 | 9.96 |
| `three_ad_legacy_cutoff` | Table 18: beta=(1,2,3,4) | `err_rL2` | 10 | 2.730e-10 | 4.387e-10 | 6.82e-11 - 9.38e-10 | 1.61 |

## Interpretation

Compare error scales and the whole distribution, rather than requiring the original printed digits. In Table 5, Nc=1920 is smaller than the 2560 output coefficients, so random feature draws can produce especially wide error ranges. Three-run architecture checks confirm the configured models and show this variability; the provided table sweep uses ten runs per cell.

The smaller cosine case in Table 22 (Nc=1920, m=640) did not reproduce the published mean in this environment. Two independent ten-run batches (`cosine` and `cosine_repeat_batch2`) are shown; their pooled twenty-run mean is about 1.455e-3 versus 1.540e-4 in the paper. This solver calls scipy.linalg.lstsq, so the NumPy cutoff option does not affect it. The dense case (Nc=10800, m=1280) is checked separately and agrees closely. Both the discrepancy and the close dense-case result are retained; agreement for every table cell is not claimed.

The AD comparison is sensitive to NumPy's numerical rank cutoff. NumPy 2 uses machine epsilon times max(M,N) by default; versions before 2.0 used machine epsilon. The `three_ad` row keeps the installed default, while `three_ad_legacy_cutoff` explicitly uses rcond=-1. The original solver source remains unchanged. Table 18's configuration selects the historical cutoff explicitly. This setting does not change SciPy solves.

## Rerunning

```bash
python scripts/check_paper_accuracy.py --case flower
python scripts/check_paper_accuracy.py --case circle_2l_sin_sin
python scripts/check_paper_accuracy.py --case three_ad_legacy_cutoff
python scripts/check_paper_accuracy.py
```

The commands save fresh measurements to `results/paper_accuracy_latest.csv` and per-case run directories. The checked-in [summary](../results/paper_accuracy_reference.csv) and [all individual runs](../results/paper_accuracy_runs.csv) record the measurements above. The input configurations are in `configs/accuracy_checks.json`.
