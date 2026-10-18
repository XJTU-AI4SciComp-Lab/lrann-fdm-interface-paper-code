"""Compare small-run fingerprints with the original Windows CPU reference."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import scipy.linalg
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrann_interface.runner import call_solver, load_experiment, main_arguments


def array_digest(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def fingerprint(path):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(2026)
    np.random.seed(2026)
    random.seed(2026)
    calls = []
    original_np = np.linalg.lstsq
    original_sp = scipy.linalg.lstsq
    original_show = plt.show

    def record(function):
        def solve(a, b, *args, **kwargs):
            result = function(a, b, *args, **kwargs)
            calls.append({"a": array_digest(a), "b": array_digest(b), "solution": array_digest(result[0])})
            return result
        return solve

    try:
        np.linalg.lstsq = record(original_np)
        scipy.linalg.lstsq = record(original_sp)
        plt.show = lambda *args, **kwargs: plt.close("all")
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            module = load_experiment(path)
            if not hasattr(module, "main"):
                return {"symbolic_stdout": stream.getvalue()}
            keywords = main_arguments(module, [])
            for key, value in {"N1": 64, "N2": 8, "N3": 32, "n_int": 4, "m": 8, "dim": 3}.items():
                if key in keywords:
                    keywords[key] = value
            if "dim" in keywords:
                keywords.update(N1=4, m_d=2, n_int=32)
            _, state = call_solver(module, keywords)
        arrays = {}
        for name, value in state.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                arrays[name] = array_digest(value)
            elif isinstance(value, torch.nn.Module):
                arrays[name] = {key: array_digest(item) for key, item in value.state_dict().items()}
        return {"least_squares": calls, "locals": arrays}
    finally:
        np.linalg.lstsq = original_np
        scipy.linalg.lstsq = original_sp
        plt.show = original_show
        plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Check all 15 retained solvers; some include 90,000-point plotting loops")
    parser.add_argument("--script", default="experiments/ex4_square/square_interface_oscillate_solution.py")
    args = parser.parse_args()
    reference = json.loads((ROOT / "tests/fixtures/numerical_comparison.json").read_text())
    paths = list(reference["scripts"]) if args.all else [args.script]
    torch.set_num_threads(reference["environment"]["threads"])
    failures = []
    for relative in paths:
        if relative not in reference["scripts"]:
            parser.error(f"Script is not in the reference manifest: {relative}")
        values = fingerprint(ROOT / relative)
        digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        equal = digest == reference["scripts"][relative]["original_sha256"]
        print(("IDENTICAL " if equal else "DIFFERENT ") + relative, flush=True)
        if not equal:
            failures.append(relative)
    if failures:
        print("The bitwise reference is specific to its recorded Windows CPU environment; compare dependency, BLAS, and thread settings.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
