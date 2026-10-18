import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lrann_interface.finite_difference import fdux, fduxx
from lrann_interface.quadrature import gauss_points_2d, gauss_weights_2d, gauss_weights_3d
from lrann_interface.runner import ROOT, main_arguments, load_experiment, run
from lrann_interface.least_squares import numpy_cutoff


def test_difference_and_quadrature():
    points = torch.tensor([[0.25, 0.75], [-0.5, 0.125]], dtype=torch.float64)

    def polynomial(x):
        values = x[:, 0:1] ** 2 + 3 * x[:, 1:2]
        return values, values

    torch.testing.assert_close(fdux(polynomial, points, 1e-6), 2 * points[:, 0:1])
    torch.testing.assert_close(fduxx(polynomial, points, 5e-4), torch.full((2, 1), 2.0, dtype=torch.float64))
    weights = gauss_weights_2d(4)
    nodes = gauss_points_2d(4)
    assert torch.sum(weights).item() == pytest.approx(4.0)
    assert torch.sum(weights * nodes[:, 0] ** 2).item() == pytest.approx(4 / 3)
    assert torch.sum(gauss_weights_3d(4)).item() == pytest.approx(8.0)


@pytest.mark.parametrize("path", sorted((ROOT / "experiments").rglob("*.py")))
def test_all_scripts_import(path):
    module = load_experiment(path)
    if hasattr(module, "main"):
        assert callable(module.get_args)


def test_integer_oscillation_arguments():
    module = load_experiment(ROOT / "experiments/ex4_square/square_interface_oscillate_solution.py")
    assert main_arguments(module, ["-k1", "4", "-k2", "3"])["k1"] == 4
    with pytest.raises(ValueError, match="positive integer"):
        main_arguments(module, ["-k1", "2.5"])


def test_explicit_historical_numpy_cutoff():
    matrix = np.zeros((100, 2))
    matrix[0, 0] = 1.0
    matrix[1, 1] = 5e-16
    rhs = matrix @ np.ones(2)
    original = np.linalg.lstsq
    with numpy_cutoff():
        assert np.linalg.lstsq is original
        assert np.linalg.lstsq(matrix, rhs)[2] == 1
    with numpy_cutoff(-1):
        assert np.linalg.lstsq(matrix, rhs)[2] == 2
    assert np.linalg.lstsq is original


def test_paper_configuration_paths_and_table5_models():
    table = json.loads((ROOT / "configs/table_05.json").read_text())
    assert len(table["cases"]) == 24
    assert len({case["label"].split(",")[0] for case in table["cases"]}) == 6
    scripts = {path.relative_to(ROOT).as_posix() for path in (ROOT / "experiments").rglob("*.py")}
    for path in (ROOT / "configs").glob("*.json"):
        data = json.loads(path.read_text())
        configs = [data] if "script" in data else ([data["base"], *data["cases"]] if "base" in data else [case["config"] for case in data["cases"]])
        for config in configs:
            if "script" in config:
                assert config["script"] in scripts, path
            if path.name != "smoke.json" and "seed" in config:
                assert config["seed"] is None, path
    module = load_experiment(ROOT / "experiments/ex1_flower/ex1_case2/circle_2d_1l.py")
    assert main_arguments(module, ["--outer-activation", "sin"])["outer_activation"] == "sin"


def test_saved_solver_outputs_and_repeated_seed(tmp_path):
    config = json.loads((ROOT / "configs/smoke.json").read_text())
    config["save_arrays"] = True
    first = run(config, output_root=tmp_path)
    with np.load(tmp_path / "results/smoke/run_01.npz") as data:
        initial = {key: data[key].copy() for key in data.files}
    second = run(config, output_root=tmp_path)
    with np.load(tmp_path / "results/smoke/run_01.npz") as data:
        for key, value in initial.items():
            np.testing.assert_array_equal(data[key], value)
        assert np.linalg.norm(data["matrix_a"] @ data["wu"].reshape(-1, 1) - data["matrix_b"]) < np.linalg.norm(data["matrix_b"])
    assert first[0]["err_rL2"] == second[0]["err_rL2"]
    assert (tmp_path / "figures/smoke/figure_08.png").stat().st_size > 1000
    assert (tmp_path / "figures/smoke/figure_08.pdf").stat().st_size > 1000
