# Experiments

The `ex1_flower/` through `ex6_moving_interface/` directories contain 15 solvers for the paper's six examples. They include the mixed flower and sphere formulations, all Table 5 architectures, the three-interface AD comparison, and the cosine-activation square problem. Unrelated parameter studies, residual networks, symbolic scratch calculations, and duplicate variants are excluded from the public project.

Use `scripts/run_experiment.py` with a configuration under `configs/` for a recorded run. Direct execution retains the original script's entry point: this may be a single solve, ten-run average, or an initialization-range parameter study. It may show plots interactively instead of saving them.

Circle solvers are named for their actual depth and exterior activation sequence. The cosine square solver is grouped with the square example. The [script inventory](../docs/script_inventory.md) maps each retained solver to the paper and its original source. The [reproduction guide](../docs/reproduction.md) lists figure and table configurations.
