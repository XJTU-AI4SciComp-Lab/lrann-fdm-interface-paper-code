"""Export solution fields for the paper geometries after the solver returns."""

import inspect
import math

import numpy as np
import torch


def field_values(module, state, geometry, resolution=300, instant=0.0):
    x = torch.linspace(module.a, module.b, resolution)
    y = torch.linspace(module.c, module.d, resolution)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    points = torch.cat((xx.flatten()[:, None], yy.flatten()[:, None]), dim=1)
    radius = points[:, 0:1] ** 2 + points[:, 1:2] ** 2
    if geometry == "flower":
        center = torch.ones(points.shape[0], 1) * 0.02 * math.sqrt(5)
        radius = (points[:, 0:1] - center) ** 2 + (points[:, 1:2] - center) ** 2
        theta = torch.atan2(points[:, 1:2] - 0.02 * math.sqrt(5), points[:, 0:1] - 0.02 * math.sqrt(5))
        regions = torch.where(radius <= (0.4 + 0.2 * torch.sin(20 * theta)) ** 2, 1, 2)
    elif geometry in ("circle", "sphere"):
        regions = torch.where(radius <= 0.75 ** 2, 1, 2)
        if geometry == "sphere":
            points = torch.cat((points, torch.zeros(points.shape[0], 1)), dim=1)
    elif geometry == "square":
        inside = (points[:, 0:1] >= -0.5) & (points[:, 0:1] <= 0.5) & (points[:, 1:2] >= -0.5) & (points[:, 1:2] <= 0.5)
        regions = torch.where(inside, 1, 2)
    elif geometry == "three_interfaces":
        theta = torch.atan2(points[:, 1:2], points[:, 0:1])
        regions = torch.zeros_like(radius, dtype=torch.int64)
        regions[radius > 0.2 ** 2] = 1
        regions[radius > (0.5 - 0.1 * torch.cos(5 * theta)) ** 2] = 2
        regions[radius > 0.8 ** 2] = 3
    elif geometry == "moving":
        regions = torch.where(radius <= (0.5 + 0.3 * instant) ** 2, 1, 2)
        points = torch.cat((points, instant * torch.ones(points.shape[0], 1)), dim=1)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    predicted = torch.zeros(points.shape[0], 1)
    exact = torch.zeros_like(predicted)
    # Keep the original one-point network evaluation order and matrix sizes.
    with torch.no_grad():
        for index in range(points.shape[0]):
            region = int(regions[index].item())
            model = state[f"model{region}"]
            solution = getattr(module, f"u{region}")
            point = points[index:index + 1]
            predicted[index, 0] = model(point)[1].reshape(-1)[0]
            arguments = [point]
            if len(inspect.signature(solution).parameters) == 2:
                arguments.append(state[f"k{region}"])
            exact[index, 0] = solution(*arguments).reshape(-1)[0]
    shape = (resolution, resolution)
    return predicted.numpy().reshape(shape), exact.numpy().reshape(shape)


def save_paper_fields(module, state, settings, directory):
    from matplotlib import pyplot as plt

    snapshots = settings.get("snapshots")
    if snapshots is None:
        snapshots = [{"number": settings["number"], "time": 0.0}]
    for item in snapshots:
        predicted, exact = field_values(module, state, settings["geometry"], settings.get("resolution", 300), item.get("time", 0.0))
        error = np.abs(predicted - exact)
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
        for axis, values, title in zip(axes, (predicted, exact, error), (r"$u_\rho$", r"$u$", r"$|u-u_\rho|$")):
            image = axis.imshow(values, interpolation="nearest", cmap="rainbow", extent=[module.a, module.b, module.c, module.d], origin="lower", aspect=1)
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        stem = directory / f"figure_{item['number']:02d}"
        figure.savefig(stem.with_suffix(".png"), dpi=200)
        figure.savefig(stem.with_suffix(".pdf"))
        np.savez_compressed(stem.with_suffix(".npz"), predicted=predicted, exact=exact, absolute_error=error)
        plt.close(figure)
        print(f"Saved {stem.with_suffix('.png')}", flush=True)
