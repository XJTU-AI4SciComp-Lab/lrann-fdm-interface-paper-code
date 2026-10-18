"""Explicit compatibility with the historical NumPy least-squares cutoff."""

from contextlib import contextmanager

import numpy as np


@contextmanager
def numpy_cutoff(rcond=None):
    """Leave calls untouched unless a cutoff is explicitly requested."""
    if rcond is None:
        yield
        return
    original = np.linalg.lstsq

    def solve(a, b, rcond=None):
        return original(a, b, rcond=cutoff if rcond is None else rcond)

    cutoff = rcond
    np.linalg.lstsq = solve
    try:
        yield
    finally:
        np.linalg.lstsq = original
