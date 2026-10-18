"""Original tensor-product Gauss-Legendre rules and ordering."""

from itertools import product

from numpy.polynomial.legendre import leggauss
import torch


def gauss_points_2d(N):
    points, weights = leggauss(N)
    l = []
    for item in product(points, points):
        l.append(list(item))
    xy = torch.tensor(l)
    return xy


def gauss_weights_2d(N):
    points, weights = leggauss(N)
    l = []
    for item in product(weights, weights):
        l.append(list(item))
    xy = torch.tensor(l)
    gz = torch.mul(xy[:, 0], xy[:, 1])
    return gz


def gauss_points_3d(N):
    points, weights = leggauss(N)
    l = []
    for item in product(points, points, points):
        l.append(list(item))
    xy = torch.tensor(l)
    return xy


def gauss_weights_3d(N):
    points, weights = leggauss(N)
    l = []
    for item in product(weights, weights, weights):
        l.append(list(item))
    xy = torch.tensor(l)
    gz = torch.mul(torch.mul(xy[:, 0], xy[:,1]), xy[:,2])
    return gz
