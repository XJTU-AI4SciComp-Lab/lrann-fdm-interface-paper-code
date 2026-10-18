"""LRaNN-FDM paper experiment; see docs/reproduction.md for settings."""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
import lrann_interface.compat

# Solve Two-dimensional Multiple Interfaces Problem (example4.3)

import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.polynomial.legendre import leggauss
from itertools import product
import argparse
import math
import time

# torch.manual_seed(20)

# define FNN
class dnn(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)                          # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)                         # output layer
        self.phi = torch.tanh                                                                  # activation function

    def forward(self, x):
        x = self.phi(self.hidden(x))                                                           # output of hidden layer
        y = self.predict(x)                                                                    # linear output
        return x, y


# parameter initialization
def weights_init1(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, ra1, rb1)
        if m.bias is not None:
            nn.init.uniform_(m.bias, ra1, rb1)


def weights_init2(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, ra2, rb2)
        if m.bias is not None:
            nn.init.uniform_(m.bias, ra2, rb2)


def weights_init3(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, ra3, rb3)
        if m.bias is not None:
            nn.init.uniform_(m.bias, ra3, rb3)


def weights_init4(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, ra4, rb4)
        if m.bias is not None:
            nn.init.uniform_(m.bias, ra4, rb4)


# approximating derivative by finite difference
# model: function, xr: sampling points, h: difference step, xn: direction
# second-derivative with respect to x
from lrann_interface.finite_difference import fduxx as fduxx


# second-derivative with respect to y
from lrann_interface.finite_difference import fduyy as fduyy


# first-derivative with respect to x
from lrann_interface.finite_difference import fdux as fdux


# first-derivative with respect to y
from lrann_interface.finite_difference import fduy as fduy


# # directional derivative
# def fdd1(model, xr, xn, h):
#     du = (model(xr+h*xn)[0] - model(xr-h*xn)[0])/(2*h)
#     return du
#
#
# def fdd2(model, xr, xn, h):
#     du = (model(xr+h*xn) - model(xr-h*xn))/(2*h)
#     return du


# Gauss-Legendre numerical integration(2-D)
# integration points
from lrann_interface.quadrature import gauss_points_2d as get_GL_points


# integration weights
from lrann_interface.quadrature import gauss_weights_2d as get_weights


# sampling points
# boundary points
def get_boundary_points0(N):
    index1 = (b-a)*torch.rand(N, 1) + a
    index2 = (d-c)*torch.rand(N, 1) + c
    xb1 = torch.cat((index1, c*torch.ones_like(index2)), dim=1)
    xb2 = torch.cat((index1, d*torch.ones_like(index2)), dim=1)
    xb3 = torch.cat((a * torch.ones_like(index1),index2), dim=1)
    xb4 = torch.cat((b * torch.ones_like(index1),index2), dim=1)
    xb = torch.cat((xb1,xb2,xb3,xb4), dim=0)
    return xb


# interior points
def get_interior_point(N):
    index1 = (b-a)*torch.rand(N, 1) + a
    index2 = (d-c)*torch.rand(N, 1) + c
    xb = torch.cat((index1, index2), dim=1)
    return xb


# interface points
def get_interface1_point(N):
    theta = 2*np.pi*torch.rand(N, 1)
    index1 = 0.2*torch.cos(theta)
    index2 = 0.2*torch.sin(theta)
    xb = torch.cat((index1, index2), dim=1)
    return xb, theta


def get_interface2_point(N):
    theta = 2*np.pi*torch.rand(N, 1)
    index1 = (0.5-0.1*torch.cos(5*theta))*torch.cos(theta)
    index2 = (0.5-0.1*torch.cos(5*theta))*torch.sin(theta)
    xb = torch.cat((index1, index2), dim=1)
    return xb, theta


def get_interface3_point(N):
    theta = 2*np.pi*torch.rand(N, 1)
    index1 = 0.8*torch.cos(theta)
    index2 = 0.8*torch.sin(theta)
    xb = torch.cat((index1, index2), dim=1)
    return xb, theta


# real solution
# solution in Omega1
def u0(x):
    return torch.cos(x[:, 1:2]) + 1.8


# solution in Omega2
def u1(x):
    return torch.exp(x[:, 0:1]) + 1.3


# solution in Omega3
def u2(x):
    return torch.sin(x[:, 0:1]) + 0.5


# partial derivative of real solution
def ux0(x):
    return torch.zeros(x.shape[0], 1)


def ux1(x):
    return torch.exp(x[:, 0:1])


def ux2(x):
    return torch.cos(x[:, 0:1])


def ux3(x):
    return -torch.ones(x.shape[0], 1)


def uy0(x):
    return -torch.sin(x[:, 1:2])


def uy1(x):
    return torch.zeros(x.shape[0], 1)


def uy2(x):
    return torch.zeros(x.shape[0], 1)


def uy3(x):
    return 1/(x[:, 1:2]+2)


# solution in Omega4
def u3(x):
    return -x[:, 0:1] + torch.log(x[:, 1:2]+2)


# source term
def f0(x):
    return a0*torch.cos(x[:, 1:2])


def f1(x):
    return -a1*torch.exp(x[:, 0:1])


def f2(x):
    return a2*torch.sin(x[:, 0:1])


def f3(x):
    return a3*torch.pow((x[:, 1:2]+2), -2)


# parameters
def get_args():
    parser = argparse.ArgumentParser(description='Train the pde net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r1', '--range1',  default=1.1, help='Range of initial weights', type=float)
    parser.add_argument('-r2', '--range2', default=0.7, help='Range of initial weights', type=float)
    parser.add_argument('-r3', '--range3',  default=0.3, help='Range of initial weights', type=float)
    parser.add_argument('-r4', '--range4', default=1, help='Range of initial weights', type=float)
    parser.add_argument('-hx1', '--difference1', default=1e-6, help='difference parameter', type=float)
    parser.add_argument('-hx2', '--difference2', default=5e-4, help='difference parameter', type=float)
    parser.add_argument('-n1', '--numbers1', default=1600, help='numbers of the interior points', type=int)
    parser.add_argument('-n2', '--numbers2', default=40, help='number of the boundary points', type=int)
    parser.add_argument('-n3', '--numbers3', default=160, help='number of the interface points', type=int)
    parser.add_argument('-n_int', '--numbers4', default=20, help='number of the integration points', type=int)
    parser.add_argument('-m', '--neurons', default=320, help='number of hidden layer nodes', type=int)
    parser.add_argument('-p1', '--penalty1', default=100, help='penalty parameter on boundary', type=float)
    parser.add_argument('-p2', '--penalty2', default=100, help='penalty parameter on interface', type=float)
    parser.add_argument('-alpha0', '--parameter1', default=0.000001, help='coefficient1 of solution', type=float)
    parser.add_argument('-alpha1', '--parameter2', default=0.0001, help='coefficient2 of solution', type=float)
    parser.add_argument('-alpha2', '--parameter3', default=0.01, help='coefficient1 of solution', type=float)
    parser.add_argument('-alpha3', '--parameter4', default=1, help='coefficient2 of solution', type=float)
    return parser.parse_args()


def main(r1, r2, r3, r4, hx1, hx2, N1, N2, N3, n_int, m, p1, p2, alpha0, alpha1, alpha2, alpha3, device):

    global ra1
    global rb1
    global ra2
    global rb2
    global ra3
    global rb3
    global ra4
    global rb4
    ra1 = -r1
    rb1 = r1
    ra2 = -r2
    rb2 = r2
    ra3 = -r3
    rb3 = r3
    ra4 = -r4
    rb4 = r4
    in_N = 2
    out_N = 1
    model0 = dnn(in_N, m, out_N).to(device)                    # approximate real solution in Omega1
    model0 = model0.double()
    model0.apply(weights_init1)

    model1 = dnn(in_N, m, out_N).to(device)                    # approximate real solution in Omega2
    model1 = model1.double()
    model1.apply(weights_init2)

    model2 = dnn(in_N, m, out_N).to(device)                    # approximate real solution in Omega3
    model2 = model2.double()
    model2.apply(weights_init3)

    model3 = dnn(in_N, m, out_N).to(device)                    # approximate real solution in Omega4
    model3 = model3.double()
    model3.apply(weights_init4)
    torch.set_default_tensor_type(torch.DoubleTensor)

    global a                                                   # domain(a,b) x (c,d)
    global b
    global c
    global d
    global h1
    global h2
    global a0
    global a1
    global a2
    global a3
    a = -1
    b = 1
    c = -1
    d = 1
    a0 = alpha0
    a1 = alpha1
    a2 = alpha2
    a3 = alpha3

    start= time.perf_counter()

    # interior euqations
    xy_inter = get_interior_point(N1)
    xy_inter = xy_inter.to(device)
    A = np.zeros((N1, 4*m))
    fv = np.zeros((N1, 1))
    test = xy_inter[:, 0:1]**2 + xy_inter[:, 1:2]**2
    theta_1 = torch.atan2(xy_inter[:,1:2], xy_inter[:, 0:1])
    laplace1 = -a0*(fduxx(model0, xy_inter, hx2)+fduyy(model0, xy_inter, hx2))
    laplace2 = -a1*(fduxx(model1, xy_inter, hx2)+fduyy(model1, xy_inter, hx2))
    laplace3 = -a2*(fduxx(model2, xy_inter, hx2)+fduyy(model2, xy_inter, hx2))
    laplace4 = -a3*(fduxx(model3, xy_inter, hx2)+fduyy(model3, xy_inter, hx2))
    for i in range(N1):
        if test[i] <= 0.2**2:
            A[i,0:m] = (laplace1[i:i+1, :]).detach().cpu().numpy()
            fv[i,:] = f0(xy_inter[i:i+1,:]).cpu().numpy()
        if 0.2**2 < test[i] <= (0.5 - 0.1 * torch.cos(5 * theta_1[i]))**2:
            A[i,m:2*m] = (laplace2[i:i+1, :]).detach().cpu().numpy()
            fv[i,:] = f1(xy_inter[i:i+1,:]).cpu().numpy()
        if (0.5 - 0.1 * torch.cos(5 * theta_1[i]))**2 < test[i] <= 0.8**2:
            A[i,2*m:3*m] = (laplace3[i:i+1, :]).detach().cpu().numpy()
            fv[i,:] = f2(xy_inter[i:i+1,:]).cpu().numpy()
        if test[i] > 0.8**2:
            A[i,3*m:4*m] = (laplace4[i:i+1, :]).detach().cpu().numpy()
            fv[i,:] = f3(xy_inter[i:i+1,:]).cpu().numpy()

    # interface1 couple
    xy_interface1 = get_interface1_point(N3)[0]
    theta_interface1 = get_interface1_point(N3)[1]
    xy_interface1 = xy_interface1.to(device)
    n11 = torch.cos(theta_interface1)
    n12 = torch.sin(theta_interface1)
    n1 = torch.cat((n11, n12), dim=1)

    C1 = np.zeros((N3,4*m))
    fc1 = np.zeros((N3,1))
    C1[:,0:m] = model0(xy_interface1)[0].detach().cpu().numpy()
    C1[:,m:2*m] = -model1(xy_interface1)[0].detach().cpu().numpy()
    fc1[:,0:1] = (u0(xy_interface1) - u1(xy_interface1)).cpu().numpy()

    D1 = np.zeros((N3,4*m))
    fd1 = np.zeros((N3,1))
    left_jump11 = a0*(fdux(model0,xy_interface1,hx1)*n11 + fduy(model0,xy_interface1,hx1)*n12)
    left_jump12 = -a1*(fdux(model1,xy_interface1,hx1)*n11 + fduy(model1,xy_interface1,hx1)*n12)
    right_jump11 = a0*(ux0(xy_interface1)*n11 + uy0(xy_interface1)*n12)
    right_jump12 = - a1*(ux1(xy_interface1)*n11 + uy1(xy_interface1)*n12)
    D1[:, 0:m] = left_jump11.detach().cpu().numpy()
    D1[:, m:2*m] = left_jump12.detach().cpu().numpy()
    fd1[:, 0:1] = (right_jump11 + right_jump12).cpu().numpy()

    # interface2 couple
    xy_interface2 = get_interface2_point(N3)[0]
    theta_interface2 = get_interface2_point(N3)[1]
    xy_interface2 = xy_interface2.to(device)

    # calculation of normal vector
    rr = 0.5 - 0.1*torch.cos(5*theta_interface2)
    rrd = 0.5*torch.sin(5*theta_interface2)
    h = torch.pow(rr**2 + rrd**2, -1/2)
    n21 = -h*(rrd*torch.sin(theta_interface2)+rr*torch.cos(theta_interface2))
    n22 = h*(rrd*torch.cos(theta_interface2)-rr*torch.sin(theta_interface2))
    n2 = torch.cat((n21, n22), dim=1)

    C2 = np.zeros((N3, 4*m))
    fc2 = np.zeros((N3, 1))
    C2[:, m:2*m] = model1(xy_interface2)[0].detach().cpu().numpy()
    C2[:, 2*m:3*m] = -model2(xy_interface2)[0].detach().cpu().numpy()
    fc2[:, 0:1] = (u1(xy_interface2) - u2(xy_interface2)).cpu().numpy()

    D2 = np.zeros((N3,4*m))
    fd2 = np.zeros((N3,1))
    left_jump21 = a1*(fdux(model1,xy_interface2,hx1)*n21 + fduy(model1,xy_interface2,hx1)*n22)
    left_jump22 = -a2*(fdux(model2,xy_interface2,hx1)*n21 + fduy(model2,xy_interface2,hx1)*n22)
    right_jump21 = a1*(ux1(xy_interface2)*n21 + uy1(xy_interface2)*n22)
    right_jump22 = - a2*(ux2(xy_interface2)*n21 + uy2(xy_interface2)*n22)
    D2[:,m:2*m] = left_jump21.detach().cpu().numpy()
    D2[:, 2*m:3*m] = left_jump22.detach().cpu().numpy()
    fd2[:,0:1] = (right_jump21 + right_jump22).cpu().numpy()
    # D2[:,m:2*m] = a1*fdd1(model1, xy_interface2, n2, hx1).detach().cpu().numpy()
    # D2[:, 2*m:3*m] = -a2*fdd1(model2, xy_interface2, n2, hx1).detach().cpu().numpy()
    # fd2[:,0:1] = (a1*fdd2(u1, xy_interface2, n2, hx1) - a2*fdd2(u2, xy_interface2, n2, hx1)).cpu().numpy()

    # interface3 couple
    xy_interface3 = get_interface3_point(N3)[0]
    theta_interface3 = get_interface3_point(N3)[1]
    xy_interface3 = xy_interface3.to(device)

    n31 = torch.cos(theta_interface3)
    n32 = torch.sin(theta_interface3)
    n3 = torch.cat((n31, n32), dim=1)

    C3 = np.zeros((N3,4*m))
    fc3 = np.zeros((N3,1))
    C3[:, 2*m:3*m] = model2(xy_interface3)[0].detach().cpu().numpy()
    C3[:, 3*m:4*m] = - model3(xy_interface3)[0].detach().cpu().numpy()
    fc3[:, 0:1] = (u2(xy_interface3) - u3(xy_interface3)).cpu().numpy()

    D3 = np.zeros((N3,4*m))
    fd3 = np.zeros((N3,1))
    left_jump31 = a2*(fdux(model2,xy_interface3,hx1)*n31 + fduy(model2,xy_interface3,hx1)*n32)
    left_jump32 = -a3*(fdux(model3,xy_interface3,hx1)*n31 + fduy(model3,xy_interface3,hx1)*n32)
    right_jump31 = a2*(ux2(xy_interface3)*n31 + uy2(xy_interface3)*n32)
    right_jump32 = - a3*(ux3(xy_interface3)*n31 + uy3(xy_interface3)*n32)
    D3[:, 2*m:3*m] = left_jump31.detach().cpu().numpy()
    D3[:, 3*m:4*m] = left_jump32.detach().cpu().numpy()
    fd3[:, 0:1] = (right_jump31 + right_jump32).cpu().numpy()

    # boundary equations
    B = np.zeros((4*N2,4*m))
    xb = get_boundary_points0(N2)
    xb = xb.to(device)
    B[:, 3*m:4*m] = model3(xb)[0].detach().cpu().numpy()
    fb = u3(xb).view([xb.size(0), 1]).cpu().numpy()

    matrix_a = np.vstack((A, p1*B, p2*C1, p2*D1, p2*C2, p2*D2, p2*C3, p2*D3))
    matrix_b = np.vstack((fv, p1*fb, p2*fc1, p2*fd1, p2*fc2, p2*fd2, p2*fc3, p2*fd3))
    wu = np.linalg.lstsq(matrix_a, matrix_b)[0]
    wu = torch.from_numpy(wu)
    wu = wu.to(device)
    wu = torch.transpose(wu, 0, 1)
    wu0 = wu[0, 0:m]
    wu1 = wu[0, m:2*m]
    wu2 = wu[0, 2*m:3*m]
    wu3 = wu[0, 3*m:4*m]

    for name, para in model0.named_parameters():
        if 'predict.weight' in name:
            para.data = wu0
    for name, para in model1.named_parameters():
        if 'predict.weight' in name:
            para.data = wu1
    for name, para in model2.named_parameters():
        if 'predict.weight' in name:
            para.data = wu2
    for name, para in model3.named_parameters():
        if 'predict.weight' in name:
            para.data = wu3

    end = time.perf_counter()
    ttt = end - start
    print('Running time: %s Seconds' % (ttt))

    # calculation of L2-error
    xr = get_GL_points(n_int)
    wr = get_weights(n_int)
    wr = wr.reshape([n_int ** 2, 1])
    wr = wr.to(device)
    wr.requires_grad_()
    xr = xr.to(device)
    xr.requires_grad_()
    x = torch.zeros_like(xr)
    x[:, 0] = (b - a) / 2 * xr[:, 0] + (a + b) / 2
    x[:, 1] = (d - c) / 2 * xr[:, 1] + (c + d) / 2
    test2 = x[:,0:1]**2 + x[:,1:2]**2
    theta_2 = torch.atan2((x[:,1:2]), x[:,0:1])
    output_u = torch.zeros(n_int**2,1)
    u_exact = torch.zeros(n_int**2,1)
    for i in range(n_int**2):
        if test2[i] <= 0.2**2:
            output_u[i:i+1,0:1] = model0(x[i:i+1,:])[1]
            u_exact[i:i+1,0:1] = u0(x[i:i+1,:])
        if test2[i] > 0.2**2 and test2[i] <= (0.5-0.1*torch.cos(5*theta_2[i]))**2:
            output_u[i:i+1,0:1] = model1(x[i:i+1,:])[1]
            u_exact[i:i+1,0:1] = u1(x[i:i+1,:])
        if test2[i] > (0.5-0.1*torch.cos(5*theta_2[i]))**2 and test2[i] <= 0.8**2:
            output_u[i:i+1,0:1] = model2(x[i:i+1,:])[1]
            u_exact[i:i+1,0:1] = u2(x[i:i+1,:])
        if test2[i] > 0.8**2:
            output_u[i:i+1,0:1] = model3(x[i:i+1,:])[1]
            u_exact[i:i+1,0:1] = u3(x[i:i+1,:])
    err_L2 = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(torch.pow(output_u - u_exact, 2) * wr))
    u_L2 = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(torch.pow(u_exact, 2) * wr))
    err_rL2 = err_L2 / u_L2
    print('Final L^2 error:', err_L2.item())
    print('Final u_L^2 norm:', u_L2.item())
    print('Final L^2 relative error:', err_rL2.item())
    return err_rL2, ttt


    # # pictures of numerical solution, real solution and error
    # with torch.no_grad():
    #     x1 = torch.linspace(a, b, 300)
    #     x2 = torch.linspace(c, d, 300)
    #     Y, X = torch.meshgrid(x2, x1)  # indexing='xy' or 'ij'
    #     Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    #     Z = Z.to(device)
    #     pred1 = torch.zeros(90000, 1)
    #     pred2 = torch.zeros(90000, 1)
    #     pred3 = torch.zeros(90000, 1)
    #     test3 = Z[:,0:1] ** 2 + Z[:,1:2] ** 2
    #     theta_3 = torch.atan2(Z[:, 1:2], Z[:, 0:1])
    #     for i in range(90000):
    #         if test3[i] <= 0.2**2:
    #             pred1[i, 0] = model0(Z[i:i+1,:])[1]
    #             pred2[i, 0] = u0(Z[i:i+1,:])
    #             pred3[i, 0] = torch.abs(model0(Z[i:i+1,:])[1]-u0(Z[i:i+1,:]))
    #         if 0.2**2 < test3[i] <= (0.5 - 0.1 * torch.cos(5 * theta_3[i]))**2:
    #             pred1[i, 0] = model1(Z[i:i+1,:])[1]
    #             pred2[i, 0] = u1(Z[i:i+1,:])
    #             pred3[i, 0] = torch.abs(model1(Z[i:i + 1, :])[1] - u1(Z[i:i + 1, :]))
    #         if (0.5 - 0.1 * torch.cos(5 * theta_3[i]))**2 < test3[i] <= 0.8**2:
    #             pred1[i, 0] = model2(Z[i:i+1,:])[1]
    #             pred2[i, 0] = u2(Z[i:i+1,:])
    #             pred3[i, 0] = torch.abs(model2(Z[i:i + 1, :])[1] - u2(Z[i:i + 1, :]))
    #         if test3[i] > 0.8**2:
    #             pred1[i, 0] = model3(Z[i:i+1,:])[1]
    #             pred2[i, 0] = u3(Z[i:i+1,:])
    #             pred3[i, 0] = torch.abs(model3(Z[i:i + 1, :])[1] - u3(Z[i:i + 1, :]))
    #
    #
    # plt.figure()
    # pred1 = pred1.cpu().numpy()
    # pred1 = pred1.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred1, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    #
    #
    # plt.figure()
    # pred2 = pred2.cpu().numpy()
    # pred2 = pred2.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred2, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    #
    #
    # plt.figure()
    # pred3 = pred3.cpu().numpy()
    # pred3 = pred3.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred3, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)


# if __name__ == '__main__':
#     args = get_args()  # Change parameters
#     device = torch.device('cpu')
#     data = np.zeros((11,1))
#     aaa = 0
#     main(r1=args.range1, r2=args.range2,r3=args.range3, r4=args.range4, hx1=args.difference1, hx2=args.difference2, N1=args.numbers1, N2=args.numbers2, N3=args.numbers3, n_int = args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha0=args.parameter1, alpha1=args.parameter2, alpha2=args.parameter3,alpha3=args.parameter4,device=device)


# #Repeat runs
if __name__ == '__main__':
    args = get_args()  # Change parameters
    device = torch.device('cpu')
    data = np.zeros((11, 1))
    ti = np.zeros((11,1))
    aaa = 0
    bbb = 0
    for i in range(10):
        data[i,0],ti[i,0] = main(r1=args.range1, r2=args.range2,r3=args.range3, r4=args.range4, hx1=args.difference1, hx2=args.difference2, N1=args.numbers1, N2=args.numbers2, N3=args.numbers3, n_int = args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha0=args.parameter1, alpha1=args.parameter2, alpha2=args.parameter3,alpha3=args.parameter4,device=device)
        aaa += data[i,0]
        bbb += ti[i,0]
    data[10,0] = aaa/10
    ti[10,0] = bbb/10
    print('%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e ' % (data[0,0],data[1,0],data[2,0],data[3,0],data[4,0],data[5,0],data[6,0],data[7,0],data[8,0],data[9,0],data[10,0]))
    print('\n')
    print('%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e ' % (ti[0,0],ti[1,0],ti[2,0],ti[3,0],ti[4,0],ti[5,0],ti[6,0],ti[7,0],ti[8,0],ti[9,0],ti[10,0]))
#     # print(data)