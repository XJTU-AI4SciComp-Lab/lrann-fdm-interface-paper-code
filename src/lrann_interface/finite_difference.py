"""Original finite-difference expressions, grouped without algebraic changes."""

import torch


def fduxx(model, xr, h):
    ei = torch.zeros_like(xr)
    ei[:, 0] = 1
    uxx = (model(xr+h*ei)[0] + model(xr-h*ei)[0] - 2*model(xr)[0])/(h**2)
    return uxx


def fduyy(model, xr, h):
    ei = torch.zeros_like(xr)
    ei[:, 1] = 1
    uyy = (model(xr+h*ei)[0] + model(xr-h*ei)[0] - 2*model(xr)[0])/(h**2)
    return uyy


def fdux(model, xr, h):
    ei = torch.zeros_like(xr)
    ei[:, 0] = 1
    ux = (model(xr+h*ei)[0] - model(xr-h*ei)[0])/(2*h)
    return ux


def fduy(model, xr, h):
    ei = torch.zeros_like(xr)
    ei[:, 1] = 1
    uy = (model(xr+h*ei)[0] - model(xr-h*ei)[0])/(2*h)
    return uy


def fdux_variant2(model, xr, h):
    ei = torch.zeros_like(xr)
    ei[:,0]=1
    uxx = (model(xr+h*ei)[0] - model(xr-h*ei)[0])/(2*h)
    return uxx


def fduy_variant2(model, xr, h):
    ei = torch.zeros_like(xr)
    ei[:,1]=1
    uyy = (model(xr+h*ei)[0] - model(xr-h*ei)[0])/(2*h)
    return uyy


def fdd1(model, xr, xn, h):
    du = (model(xr+h*xn)[0] - model(xr-h*xn)[0])/(2*h)
    return du


def fdd2(model, xr, xn, h):
    du = (model(xr+h*xn) - model(xr-h*xn))/(2*h)
    return du


def fdux_variant3(model, xr, h):     # Finite difference approximation of u_x
    ei = torch.zeros_like(xr)
    ei[:, 0:1] = 1
    ux = (model(xr+h*ei)[0] - model(xr-h*ei)[0])/(2*h)
    return ux


def fduzz(model, xr, h):
    ei = torch.zeros_like(xr)
    ei[:,2]=1
    uzz = (model(xr+h*ei)[0] + model(xr-h*ei)[0] - 2*model(xr)[0])/(h**2)
    return uzz


def fduz(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,2]=1
    uzz = (model(xr+h*ei)[0] - model(xr-h*ei)[0])/(2*h)
    return uzz
