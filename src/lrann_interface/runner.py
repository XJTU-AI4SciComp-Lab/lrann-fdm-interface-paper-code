"""Run the original solver functions with explicit inputs and saved artifacts."""

import argparse
import ast
import contextlib
import csv
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

from .least_squares import numpy_cutoff

ROOT = Path(__file__).resolve().parents[2]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def load_experiment(path):
    """Import an experiment without executing its historical ten-run entry point."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("_lrann_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main_arguments(module, arguments):
    """Use the script's own parser and original main-call keyword mapping."""
    original_argv = sys.argv
    sys.argv = [module.__file__, *arguments]
    try:
        args = module.get_args()
    finally:
        sys.argv = original_argv
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "main"
    )
    keywords = {}
    for value in call.args:
        if isinstance(value, ast.Name) and value.id in ("r1", "r2"):
            keywords[value.id] = getattr(args, "range" + value.id[1:])
        else:
            raise ValueError(f"Unsupported positional main argument: {ast.unparse(value)}")
    for keyword in call.keywords:
        value = keyword.value
        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "args":
            keywords[keyword.arg] = getattr(args, value.attr)
        elif isinstance(value, ast.Name) and value.id == "device":
            keywords[keyword.arg] = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported main argument: {ast.unparse(value)}")
    # Some legacy parsers declared integer oscillation counts as floats.
    for name in ("k1", "k2"):
        if name in keywords:
            value = keywords[name]
            if not float(value).is_integer() or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            keywords[name] = int(value)
    return keywords


def call_solver(module, keywords):
    """Collect artifacts at main's return without editing its calculations."""
    state = {}
    previous_profile = sys.getprofile()

    def capture(frame, event, result):
        if event == "return" and frame.f_code is module.main.__code__:
            state.update(frame.f_locals)

    sys.setprofile(capture)
    try:
        result = module.main(**keywords)
    finally:
        sys.setprofile(previous_profile)
    return result, state


def scalar_metrics(state):
    metrics = {}
    for name in ("err_L2", "u_L2", "err_rL2", "err_L2_p", "err_rL2_p", "ttt"):
        value = state.get(name)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            metrics[name] = value.detach().cpu().item()
        elif isinstance(value, (int, float)):
            metrics[name] = float(value)
    return metrics


def run(config, extra_arguments=(), output_root=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    config = dict(config)
    name = config["name"]
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("The run name must be a single directory name")
    script = (ROOT / config["script"]).resolve()
    if not script.is_relative_to(ROOT / "experiments"):
        raise ValueError("Experiment scripts must be inside experiments/")
    output_root = Path(output_root).resolve() if output_root else ROOT
    results_dir = output_root / "results" / name
    figures_dir = output_root / "figures" / name
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    config["arguments"] = [*config.get("arguments", []), *extra_arguments]
    repeats = config.get("repeats", 1)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    (results_dir / "config_used.json").write_text(json.dumps(config, indent=2) + "\n")
    environment = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "numpy", "scipy", "matplotlib"):
        environment[package] = importlib.metadata.version(package)
    (results_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    seed = config.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    saved_dtype = torch.get_default_dtype()
    saved_show = plt.show
    saved_threads = torch.get_num_threads()
    # This matches the original scripts' initial dtype in a fresh Python process.
    torch.set_default_dtype(torch.float32)
    if config.get("threads") is not None:
        torch.set_num_threads(config["threads"])
    rows = []
    panel = 0
    current_repeat = 0

    def save_open_figures(*args, **kwargs):
        nonlocal panel
        for number in plt.get_fignums():
            panel += 1
            figure = plt.figure(number)
            figure.savefig(figures_dir / f"run_{current_repeat:02d}_panel_{panel:02d}.png", dpi=150)
            plt.close(figure)

    try:
        plt.show = save_open_figures
        with (results_dir / "log.txt").open("w", encoding="utf-8") as stream:
            with contextlib.redirect_stdout(Tee(sys.stdout, stream)):
                print(f"Running {name}: {script.relative_to(ROOT)}", flush=True)
                module = load_experiment(script)
                keywords = main_arguments(module, config["arguments"])
                for current_repeat in range(1, repeats + 1):
                    started = time.perf_counter()
                    with numpy_cutoff(config.get("numpy_lstsq_rcond")):
                        _, state = call_solver(module, keywords)
                    rows.append({"run": current_repeat, **scalar_metrics(state), "wall_seconds": time.perf_counter() - started})
                    save_open_figures()
                    if config.get("save_arrays"):
                        arrays = {}
                        for key in ("matrix_a", "matrix_b", "wu", "output_u", "u_exact"):
                            value = state.get(key)
                            if isinstance(value, torch.Tensor):
                                value = value.detach().cpu().numpy()
                            if isinstance(value, np.ndarray):
                                arrays[key] = value
                        np.savez_compressed(results_dir / f"run_{current_repeat:02d}.npz", **arrays)
                    if config.get("figure") and current_repeat == repeats:
                        from .plotting import save_paper_fields
                        save_paper_fields(module, state, config["figure"], figures_dir)
                    state.clear()
    finally:
        plt.show = saved_show
        plt.close("all")
        torch.set_default_dtype(saved_dtype)
        torch.set_num_threads(saved_threads)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (results_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "summary.json").write_text(json.dumps({
        "runs": repeats,
        "means": {key: float(np.mean([row[key] for row in rows if key in row])) for key in fields if key != "run"},
    }, indent=2) + "\n")
    print(f"Results: {results_dir}", flush=True)
    return rows


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--name")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--numpy-lstsq-rcond", type=float, help="Explicit NumPy cutoff; -1 matches its pre-2.0 default")
    args, extra = parser.parse_known_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("seed", "repeats", "name"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.save_arrays:
        config["save_arrays"] = True
    if args.numpy_lstsq_rcond is not None:
        config["numpy_lstsq_rcond"] = args.numpy_lstsq_rcond
    if extra[:1] == ["--"]:
        extra = extra[1:]
    run(config, extra, args.output_root)
