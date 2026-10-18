"""LRaNN-FDM paper experiment; see docs/reproduction.md for settings."""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
import lrann_interface.compat

# Solve Parabolic Interface Problem with a Dynamic Interface

import torch
import torch.nn as nn
import numpy as np
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
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)                             # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)                            # output layer
        self.phi = torch.tanh                                                                     # activation function

    def forward(self, x):
        x = self.phi(self.hidden(x))                                                              # output of hidden layer
        y = self.predict(x)                                                                       # linear output
        return x,y


# parameter initialization
def weights_init1(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight,ra1,rb1)
        if m.bias is not None:
            nn.init.uniform_(m.bias,ra1,rb1)


def weights_init2(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight,ra2,rb2)
        if m.bias is not None:
            nn.init.uniform_(m.bias,ra2,rb2)


# approximating derivative by finite difference
# model: function, xr: sampling points, h: difference step, xn: direction
# second-derivative with respect to x
from lrann_interface.finite_difference import fduxx as fduxx


# second-derivative with respect to y
from lrann_interface.finite_difference import fduyy as fduyy


# first-derivative with respect to x
from lrann_interface.finite_difference import fdux_variant2 as fdux


# first-derivative with respect to y
from lrann_interface.finite_difference import fduy_variant2 as fduy


# first-derivative with respect to t
def fdut(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,2]=1
    utt = (model(xr+h*ei)[0] - model(xr-h*ei)[0])/(2*h)
    return utt



# Gauss-Legendre numerical integration(2-D)
# integration points
from lrann_interface.quadrature import gauss_points_3d as get_GL_points


# integration weights
from lrann_interface.quadrature import gauss_weights_3d as get_weights


# sampling points
# boundary points
def get_boundary_points0(N,TT):
    index1 = (b-a)*torch.rand(N, 1) + a
    index2 = (d-c)*torch.rand(N, 1) + c
    tt = TT*torch.rand(N,1)
    xb1 = torch.cat((index1, c*torch.ones_like(index2),tt), dim=1)
    xb2 = torch.cat((index1, d*torch.ones_like(index2),tt), dim=1)
    xb3 = torch.cat((a * torch.ones_like(index1),index2,tt), dim=1)
    xb4 = torch.cat((b * torch.ones_like(index1),index2,tt), dim=1)
    xb = torch.cat((xb1,xb2,xb3,xb4), dim=0)
    return xb


# initial points
def get_initial_points(N):
    index1 = (b - a) * torch.rand(N, 1) + a
    xb = torch.cat((index1,index1,torch.zeros((N,1))),dim=1)
    return xb


# interior points
def get_interior_point(N,TT):
    index1 = (b-a)*torch.rand(N, 1) + a
    index2 = (d-c)*torch.rand(N, 1) + c
    index3 = TT*torch.rand(N,1)
    xb = torch.cat((index1, index2,index3), dim=1)
    return xb


# interface points
def get_interface_point(N,TT):
    theta = 2*np.pi*torch.rand(N,1)
    tt = TT*torch.rand(N,1)
    index1 = (0.5 + 0.3 * tt)*torch.cos(theta)
    index2 = (0.5 + 0.3 * tt)*torch.sin(theta)
    index3 = tt
    xb = torch.cat((index1, index2,index3), dim=1)
    return xb,theta


# real solution
# solution in Omega1
def u1(x):
    return -torch.exp(-x[:,2:3])*(8*x[:,0:1]**2+8*x[:,1:2]**2-3.5)


# solution in Omega2
def u2(x):
    return torch.exp(x[:,0:1]-x[:,2:3])*torch.cos(np.pi*x[:,1:2]*0.5)


def u1x(x):
    return -16*x[:,0:1]*torch.exp(-x[:,2:3])


def u2x(x):
    return torch.exp(x[:,0:1]-x[:,2:3])*torch.cos(np.pi*x[:,1:2]*0.5)


def u1y(x):
    return -16*x[:,1:2]*torch.exp(-x[:,2:3])


def u2y(x):
    return -0.5*np.pi*torch.exp(x[:,0:1]-x[:,2:3])*torch.sin(np.pi*x[:,1:2]*0.5)

# source term
def f1(x):
    return torch.exp(-x[:,2:3])*(8*x[:,0:1]**2+8*x[:,1:2]**2+32*a1-3.5)


def f2(x):
    return (0.25*a2*np.pi**2-a2-1)*torch.exp(x[:,0:1]-x[:,2:3])*torch.cos(np.pi*x[:,1:2]*0.5)


# parameters
def get_args():
    parser = argparse.ArgumentParser(description='Train the pde net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r1', '--range1',  default=1, help='Range of initial weights', type=float)
    parser.add_argument('-r2', '--range2', default=1, help='Range of initial weights', type=float)
    parser.add_argument('-hx1', '--difference1', default=1e-6, help='difference parameter', type=float)
    parser.add_argument('-hx2', '--difference2', default=5e-4, help='difference parameter', type=float)
    parser.add_argument('-n1', '--numbers1', default=1000, help='numbers of the interior points', type=int)
    parser.add_argument('-n2', '--numbers2', default=100, help='number of the boundary points', type=int)         # N**2
    parser.add_argument('-n3', '--numbers3', default=500, help='number of the interface points', type=int)        # 5*N**2
    parser.add_argument('-n_int', '--numbers4', default=20, help='number of the integration points', type=int)
    parser.add_argument('-m', '--neurons', default=640, help='number of hidden layer nodes', type=int)
    parser.add_argument('-T', '--time', default=1, help='observation time', type=float)
    parser.add_argument('-p1', '--penalty1', default=100, help='penalty parameter on boundary', type=float)
    parser.add_argument('-p2', '--penalty2', default=100, help='penalty parameter on interface', type=float)
    parser.add_argument('-p3', '--penalty3', default=100, help='penalty parameter on initial', type=float)
    parser.add_argument('-alpha1', '--parameter1', default=100, help='coefficient1 of solution', type=float)
    parser.add_argument('-alpha2', '--parameter2', default=1, help='coefficient2 of solution', type=float)
    return parser.parse_args()


def main(r1, r2, hx1, hx2, N1, N2, N3, n_int, m, T, p1, p2, p3, alpha1, alpha2, device):
    global ra1
    global rb1
    global ra2
    global rb2
    ra1 = -r1
    rb1 = r1
    ra2 = -r2
    rb2 = r2
    in_N = 3
    out_N = 1

    model1 = dnn(in_N, m, out_N).to(device)                              # approximate real solution in Omega1
    model1 = model1.double()
    model1.apply(weights_init1)

    model2 = dnn(in_N, m, out_N).to(device)                              # approximate real solution in Omega2
    model2 = model2.double()
    model2.apply(weights_init2)
    torch.set_default_tensor_type(torch.DoubleTensor)

    global a                                                             # domain(a,b) x (c,d)
    global b
    global c
    global d
    global a1
    global a2
    a = -1
    b = 1
    c = -1
    d = 1
    a1 = alpha1
    a2 = alpha2

    start = time.perf_counter()

    # interior euqations
    xy_inter = get_interior_point(N1,T)
    xy_inter = xy_inter.to(device)
    A = np.zeros((N1,2*m))
    fv = np.zeros((N1,1))
    test = (xy_inter[:,0:1])**2 + (xy_inter[:,1:2])**2
    left1 = fdut(model1, xy_inter, hx1) - a1*(fduxx(model1, xy_inter, hx2) + fduyy(model1, xy_inter, hx2))
    left2 = fdut(model2, xy_inter, hx1) - a2*(fduxx(model2, xy_inter, hx2) + fduyy(model2, xy_inter, hx2))
    for i in range(N1):
        if test[i] <= (0.5 + 0.3*xy_inter[i:i+1,2])**2:
            A[i,0:m] = (left1[i:i+1, :]).detach().cpu().numpy()
            fv[i,:] = f1(xy_inter[i:i+1,:]).cpu().numpy()
        else:
            A[i,m:2*m] = (left2[i:i+1, :]).detach().cpu().numpy()
            fv[i,:] = f2(xy_inter[i:i+1,:]).cpu().numpy()

    # interface couple
    xy_interface = get_interface_point(N3,T)[0]
    theta_interface = get_interface_point(N3,T)[1]
    xy_interface = xy_interface.to(device)

    n1 = torch.cos(theta_interface)
    n2 = torch.sin(theta_interface)
    n3 = torch.zeros(N3,1)
    n = torch.cat((n1, n2, n3), dim=1)

    C = np.zeros((N3,2*m))
    fc = np.zeros((N3,1))
    uh_jump1 = model1(xy_interface)[0]
    uh_jump2 = -model2(xy_interface)[0]
    C[:,0:m] = (uh_jump1).detach().cpu().numpy()
    C[:,m:2*m] = (uh_jump2).detach().cpu().numpy()
    fc[:,0:1] = (u1(xy_interface) - u2(xy_interface)).cpu().numpy()


    D = np.zeros((N3,2*m))                                        #Include the normal vector
    fd = np.zeros((N3,1))
    left_jump1 = a1*(fdux(model1,xy_interface,hx1)*n1 + fduy(model1,xy_interface,hx1)*n2)
    left_jump2 = -a2*(fdux(model2,xy_interface,hx1)*n1 + fduy(model2,xy_interface,hx1)*n2)
    right_jump1 = a1*(u1x(xy_interface)*n1 + u1y(xy_interface)*n2)
    right_jump2 = -a2*(u2x(xy_interface)*n1 + u2y(xy_interface)*n2)
    D[:,0:m] = left_jump1.detach().cpu().numpy()
    D[:, m:2*m] = left_jump2.detach().cpu().numpy()
    fd[:,0:1] = (right_jump1 + right_jump2).cpu().numpy()

    # boundary equations
    B = np.zeros((4*N2,2*m))
    xb = get_boundary_points0(N2,T)
    xb = xb.to(device)
    B[:,m:2*m] = (model2(xb)[0]).detach().cpu().numpy()
    fb = (u2(xb)).view([xb.size(0), 1]).cpu().numpy()

    # initial equations
    I = np.zeros((N2,2*m))
    fi = np.zeros((N2,1))
    xi = get_initial_points(N2)
    xi = xi.to(device)
    test_i = (xi[:,0:1])**2 + (xi[:,1:2])**2
    for i in range(N2):
        if test_i[i] <= 0.5**2:
            I[i,0:m] = (model1(xi[i:i+1,:])[0]).detach().cpu().numpy()
            fi[i,:] = u1(xi[i:i+1,:]).cpu().numpy()
        else:
            I[i,m:2*m] = (model2(xi[i:i+1,:])[0]).detach().cpu().numpy()
            fi[i,:] = u2(xi[i:i+1,:]).cpu().numpy()


    # solving linear system
    matrix_a = np.vstack((A, p1*B, p2*C, p2*D, p3*I))
    matrix_b = np.vstack((fv, p1*fb, p2*fc, p2*fd, p3*fi))
    wu = np.linalg.lstsq(matrix_a, matrix_b)[0]
    wu = torch.from_numpy(wu)
    wu = wu.to(device)
    wu = torch.transpose(wu, 0, 1)
    wu1 = wu[0, 0:m]
    wu2 = wu[0, m:2*m]

    for name, para in model1.named_parameters():
        if 'predict.weight' in name:
            para.data = wu1
    for name, para in model2.named_parameters():
        if 'predict.weight' in name:
            para.data = wu2

    end = time.perf_counter()
    ttt = end - start
    print('Running time: %s Seconds' % (ttt))

    # calculation of L2-error
    xr = get_GL_points(n_int)
    wr = get_weights(n_int)
    wr = wr.reshape([n_int ** 3, 1])
    wr = wr.to(device)
    wr.requires_grad_()
    xr = xr.to(device)
    xr.requires_grad_()
    x = torch.zeros_like(xr)
    x[:, 0] = (b - a) / 2 * xr[:, 0] + (a + b) / 2
    x[:, 1] = (d - c) / 2 * xr[:, 1] + (c + d) / 2
    x[:, 2] = T / 2 * xr[:, 2] + T / 2
    test2 = x[:,0:1]**2 + x[:,1:2]**2
    output_u = torch.zeros(n_int**3,1)
    u_exact = torch.zeros(n_int**3,1)
    for i in range(n_int**3):
        if test2[i] <= (0.5 + 0.3*x[i:i+1,2])**2:
            output_u[i:i+1,0:1] = model1(x[i:i+1,:])[1]
            u_exact[i:i+1,0:1] = u1(x[i:i+1,:])
        else:
            output_u[i:i+1,0:1] = model2(x[i:i+1,:])[1]
            u_exact[i:i+1,0:1] = u2(x[i:i+1,:])
    err_L2 = torch.sqrt((b - a) * (d - c) * T / 8 * torch.sum(torch.pow(output_u - u_exact, 2) * wr))
    u_L2 = torch.sqrt((b - a) * (d - c) * T / 8 * torch.sum(torch.pow(u_exact, 2) * wr))
    err_rL2 = err_L2 / u_L2
    print('Final L^2 error:', err_L2.item())
    print('Final u_L^2 norm:', u_L2.item())
    print('Final L^2 relative error:', err_rL2.item())
    return err_rL2, ttt

    # # pictures of numerical solution, real solution and error
    # with torch.no_grad():
    #     x1 = torch.linspace(a, b, 300)
    #     x2 = torch.linspace(c, d, 300)
    #     Y, X = torch.meshgrid(x2, x1)                                      # indexing='xy' or 'ij'
    #     Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    #     Z = Z.to(device)
    #     zz1 = torch.cat((Z, 0*torch.ones(90000, 1)), dim=1)
    #     zz2 = torch.cat((Z, 0.25*torch.ones(90000, 1)), dim=1)
    #     zz3 = torch.cat((Z, 0.5*torch.ones(90000, 1)), dim=1)
    #     zz4 = torch.cat((Z, 0.75*torch.ones(90000, 1)), dim=1)
    #     zz5 = torch.cat((Z, torch.ones(90000, 1)), dim=1)
    #     pred1 = torch.zeros(90000, 1)
    #     pred2 = torch.zeros(90000, 1)
    #     pred4 = torch.zeros(90000, 1)
    #     pred5 = torch.zeros(90000, 1)
    #     pred7 = torch.zeros(90000, 1)
    #     pred8 = torch.zeros(90000, 1)
    #     pred10 = torch.zeros(90000, 1)
    #     pred11 = torch.zeros(90000, 1)
    #     pred13 = torch.zeros(90000, 1)
    #     pred14 = torch.zeros(90000, 1)
    #     test3 = Z[:,0:1] ** 2 + Z[:,1:2] ** 2
    #     for i in range(90000):
    #         if test3[i] <= (0.5 + 0.3 * zz1[i:i+1,2])**2:
    #             pred1[i, 0] = model1(zz1[i:i+1,:])[1]
    #             pred2[i, 0] = u1(zz1[i:i+1,:])
    #         else:
    #             pred1[i, 0] = model2(zz1[i:i + 1, :])[1]
    #             pred2[i, 0] = u2(zz1[i:i + 1, :])
    #
    #     for i in range(90000):
    #         if test3[i] <= (0.5 + 0.3 * zz2[i:i + 1, 2]) ** 2:
    #             pred4[i, 0] = model1(zz2[i:i+1,:])[1]
    #             pred5[i, 0] = u1(zz2[i:i+1,:])
    #         else:
    #             pred4[i, 0] = model2(zz2[i:i+1,:])[1]
    #             pred5[i, 0] = u2(zz2[i:i+1,:])
    #
    #     for i in range(90000):
    #         if test3[i] <= (0.5 + 0.3 * zz3[i:i + 1, 2]) ** 2:
    #             pred7[i, 0] = model1(zz3[i:i+1,:])[1]
    #             pred8[i, 0] = u1(zz3[i:i+1,:])
    #         else:
    #             pred7[i, 0] = model2(zz3[i:i+1,:])[1]
    #             pred8[i, 0] = u2(zz3[i:i+1,:])
    #
    #     for i in range(90000):
    #         if test3[i] <= (0.5 + 0.3 * zz4[i:i + 1, 2]) ** 2:
    #             pred10[i, 0] = model1(zz4[i:i+1,:])[1]
    #             pred11[i, 0] = u1(zz4[i:i+1,:])
    #         else:
    #             pred10[i, 0] = model2(zz4[i:i+1,:])[1]
    #             pred11[i, 0] = u2(zz4[i:i+1,:])
    #
    #     for i in range(90000):
    #         if test3[i] <= (0.5 + 0.3 * zz4[i:i + 1, 2]) ** 2:
    #             pred13[i, 0] = model1(zz5[i:i+1,:])[1]
    #             pred14[i, 0] = u1(zz5[i:i+1,:])
    #         else:
    #             pred13[i, 0] = model2(zz5[i:i+1,:])[1]
    #             pred14[i, 0] = u2(zz5[i:i+1,:])
    #
    #     pred3 = torch.abs(pred1 - pred2)
    #     pred6 = torch.abs(pred4 - pred5)
    #     pred9 = torch.abs(pred7 - pred8)
    #     pred12 = torch.abs(pred10 - pred11)
    #     pred15 = torch.abs(pred13 - pred14)
    #
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
    #
    # plt.figure()
    # pred4 = pred4.cpu().numpy()
    # pred4 = pred4.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred4, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    #
    #
    # plt.figure()
    # pred5 = pred5.cpu().numpy()
    # pred5 = pred5.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred5, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    #
    #
    # plt.figure()
    # pred6 = pred6.cpu().numpy()
    # pred6 = pred6.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred6, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    #
    # plt.figure()
    # pred7 = pred7.cpu().numpy()
    # pred7 = pred7.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred7, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    #
    #
    # plt.figure()
    # pred8 = pred8.cpu().numpy()
    # pred8 = pred8.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred8, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    #
    #
    # plt.figure()
    # pred9 = pred9.cpu().numpy()
    # pred9 = pred9.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred9, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    #
    # plt.figure()
    # pred10 = pred10.cpu().numpy()
    # pred10 = pred10.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred10, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    #
    #
    # plt.figure()
    # pred11 = pred11.cpu().numpy()
    # pred11 = pred11.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred11, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    #
    #
    # plt.figure()
    # pred12 = pred12.cpu().numpy()
    # pred12 = pred12.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred12, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    #
    # plt.figure()
    # pred13 = pred13.cpu().numpy()
    # pred13 = pred13.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred13, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    #
    #
    # plt.figure()
    # pred14 = pred14.cpu().numpy()
    # pred14 = pred14.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred14, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    #
    #
    # plt.figure()
    # pred15 = pred15.cpu().numpy()
    # pred15 = pred15.reshape(300, 300)
    # ax = plt.subplot(1, 1, 1)
    # h = plt.imshow(pred15, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)


# if __name__ == '__main__':
#     args = get_args()
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
#     main(r1=args.range1, r2=args.range2, hx1=args.difference1, hx2=args.difference2, N1=args.numbers1,
#          N2=args.numbers2, N3=args.numbers3, n_int=args.numbers4, m=args.neurons, T=args.time, p1=args.penalty1, p2=args.penalty2, p3=args.penalty3, alpha1=args.parameter1, alpha2=args.parameter2, device=device)



# run ten times and give the average value
if __name__ == '__main__':
    args = get_args()  # Change parameters
    data = np.zeros((11,1))
    device = torch.device('cpu')
    ti = np.zeros((11,1))
    aaa = 0
    bbb = 0
    for i in range(10):
        data[i,0],ti[i,0] = main(r1=args.range1, r2=args.range2, hx1=args.difference1, hx2=args.difference2, N1=args.numbers1,
             N2=args.numbers2, N3=args.numbers3, n_int=args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, p3=args.penalty3, T=args.time, alpha1=args.parameter1, alpha2=args.parameter2, device=device)
        aaa += data[i, 0]
        bbb += ti[i,0]
    data[10, 0] = aaa / 10
    ti[10,0] = bbb/10
    print('%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e ' % (data[0, 0], data[1, 0], data[2, 0], data[3, 0], data[4, 0], data[5, 0], data[6, 0], data[7, 0], data[8, 0],data[9, 0], data[10, 0]))
    print('\n')
    print('%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e ' % (ti[0,0],ti[1,0],ti[2,0],ti[3,0],ti[4,0],ti[5,0],ti[6,0],ti[7,0],ti[8,0],ti[9,0],ti[10,0]))
    # print(data)



