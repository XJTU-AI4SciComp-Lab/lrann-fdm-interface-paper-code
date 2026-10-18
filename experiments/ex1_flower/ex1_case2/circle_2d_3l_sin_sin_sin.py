"""LRaNN-FDM paper experiment; see docs/reproduction.md for settings."""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / 'src'))
import lrann_interface.compat

# Solve Two-dimensional Flower Shape Interface Problem (example4.1)

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.polynomial.legendre import leggauss
from itertools import product
import argparse
import scipy.linalg
import datetime
import math
import time
import pandas as pd

# # fix random seed
# torch.manual_seed(40)

# define FNN
class dnn(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)             # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)            # output layer
        self.phi = torch.tanh                                                     # activation function

    def forward(self, x):
        x = self.phi(self.hidden(x))                                              # output of hidden layer
        y = self.predict(x)                                                       # linear output
        return x, y


class dnn3(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden, bias=True)         # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden, bias=True)         # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden, bias=True)         # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)        # output layer
        self.phi1 = torch.sin                                                # activation function
        self.phi2 = torch.sin                                                 # activation function
        self.phi3 = torch.sin                                                 # activation function

    def forward(self, x):
        x1 = self.phi1(self.hidden1(x))                                          # output of hidden layer
        x2 = self.phi2(self.hidden2(x1))                                          # output of hidden layer
        x3 = self.phi3(self.hidden3(x2))                                          # output of hidden layer
        y = self.predict(x3)                                                   # linear output
        return x3,y

# parameter initialization
def weights_init1(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, ra1, rb1)
        if m.bias is not None:
            nn.init.uniform_(m.bias, ra1, rb1)


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
    xb1 = torch.cat((index1, c*torch.ones_like(index2)), dim=1)    # bottom edge
    xb2 = torch.cat((index1, d*torch.ones_like(index2)), dim=1)    # top edge
    xb3 = torch.cat((a * torch.ones_like(index1),index2), dim=1)   # left edge
    xb4 = torch.cat((b * torch.ones_like(index1),index2), dim=1)   # right edge
    xb = torch.cat((xb1, xb2, xb3, xb4), dim=0)
    return xb


# interior points
def get_interior_point(N):
    index1 = (b-a)*torch.rand(N, 1) + a
    index2 = (d-c)*torch.rand(N, 1) + c
    xb = torch.cat((index1, index2), dim=1)
    return xb


# interface points
def get_interface_point(N):
    theta = 2*math.pi*torch.rand(N,1)
    index1 = 0.75 * torch.cos(theta)
    index2 = 0.75 * torch.sin(theta)
    xb = torch.cat((index1, index2), dim=1)
    n1 = torch.cos(theta)
    n2 = torch.sin(theta)
    return xb, n1, n2


# real solution
# solution in Omega1
def u1(x):
    return 1/a1*torch.exp(x[:, 0:1]*x[:, 1:2])


# solution in Omega2
def u2(x):
    return 1/a2*torch.mul(torch.sin(8*math.pi*x[:, 0:1]), torch.sin(8*math.pi*x[:, 1:2]))


# partial derivative of real solution
def ux1(x):
    return 1/a1*x[:,1:2]*torch.exp(x[:, 0:1]*x[:, 1:2])


def ux2(x):
    return 1/a2*8*math.pi*torch.mul(torch.cos(8*math.pi*x[:, 0:1]), torch.sin(8*math.pi*x[:, 1:2]))


def uy1(x):
    return 1/a1*x[:,0:1]*torch.exp(x[:, 0:1]*x[:, 1:2])


def uy2(x):
    return 1/a2*8*math.pi*torch.mul(torch.sin(8*math.pi*x[:, 0:1]), torch.cos(8*math.pi*x[:, 1:2]))


# source term
def f1(x):
    return -(x[:, 0]**2+x[:, 1]**2)*torch.exp(x[:, 0:1]*x[:, 1:2])


def f2(x):
    return 128*math.pi**2*torch.mul(torch.sin(8*math.pi*x[:, 0:1]), torch.sin(8*math.pi*x[:, 1:2]))


# parameters
def get_args():
    parser = argparse.ArgumentParser(description='Train the pde net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r1', '--range1',  default=1.2, help='Range of initial weights', type=float)
    parser.add_argument('-hx1', '--difference1', default=1e-6, help='difference parameter', type=float)
    parser.add_argument('-hx2', '--difference2', default=5e-4, help='difference parameter', type=float)
    parser.add_argument('-n1', '--numbers1', default=6400, help='numbers of the interior points', type=int)
    parser.add_argument('-n2', '--numbers2', default=80, help='number of one boundary points', type=int)
    parser.add_argument('-n3', '--numbers3', default=320, help='number of the interface points', type=int)
    parser.add_argument('-n_int', '--numbers4', default=20, help='number of the integration points', type=int)
    parser.add_argument('-m', '--neurons', default=1280, help='number of hidden layer nodes', type=int)
    parser.add_argument('-p1', '--penalty1', default=100, help='penalty parameter on boundary', type=float)
    parser.add_argument('-p2', '--penalty2', default=100, help='penalty parameter on boundary', type=float)
    parser.add_argument('-alpha1', '--parameter1', default=1, help='coefficient1 of solution', type=float)
    parser.add_argument('-alpha2', '--parameter2', default=1, help='coefficient2 of solution', type=float)
    return parser.parse_args()


def main(r1, hx1, hx2, N1, N2, N3, n_int, m, p1, p2, alpha1, alpha2, device):

# def get_args():
#     parser = argparse.ArgumentParser(description='Train the pde net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('-r21', '--range22', default=1.2, help='Range of initial weights', type=float)
#     parser.add_argument('-r22', '--range23', default=1.2, help='Range of initial weights', type=float)
#     parser.add_argument('-hx1', '--difference1', default=1e-6, help='difference parameter', type=float)
#     parser.add_argument('-hx2', '--difference2', default=5e-4, help='difference parameter', type=float)
#     parser.add_argument('-n1', '--numbers1', default=10000, help='numbers of the interior points', type=int)
#     parser.add_argument('-n2', '--numbers2', default=100, help='number of one boundary points', type=int)
#     parser.add_argument('-n3', '--numbers3', default=400, help='number of the interface points', type=int)
#     parser.add_argument('-n_int', '--numbers4', default=20, help='number of the integration points', type=int)
#     parser.add_argument('-m', '--neurons', default=1280, help='number of hidden layer nodes', type=int)
#     parser.add_argument('-p1', '--penalty1', default=100, help='penalty parameter on boundary', type=float)
#     parser.add_argument('-p2', '--penalty2', default=100, help='penalty parameter on boundary', type=float)
#     parser.add_argument('-alpha1', '--parameter1', default=1, help='coefficient1 of solution', type=float)
#     parser.add_argument('-alpha2', '--parameter2', default=1, help='coefficient2 of solution', type=float)
#     return parser.parse_args()
#
# def main(r22, r23, hx1, hx2, N1, N2, N3, n_int, m, p1, p2, alpha1, alpha2, device):

    global ra1
    global rb1
    ra1 = -r1
    rb1 = r1
    in_N = 2
    out_N = 1

    model1 = dnn(in_N, m, out_N).to(device)                           # approximate real solution in Omega1
    model1 = model1.double()
    model1.apply(weights_init1)


    model2 = dnn3(in_N, m, out_N).to(device)                           # approximate real solution in Omega2
    model2 = model2.double()
    r21 = 0.2
    r22 = 0.4
    r23 = 0.6
    nn.init.uniform_(model2.hidden1.weight, -r21, r21)
    nn.init.uniform_(model2.hidden1.bias, -r21, r21)
    nn.init.uniform_(model2.hidden2.weight, -r22, r22)
    nn.init.uniform_(model2.hidden2.bias, -r22, r22)
    nn.init.uniform_(model2.hidden3.weight, -r23, r23)
    nn.init.uniform_(model2.hidden3.bias, -r23, r23)
    model2.phi1 = torch.sin
    model2.phi2 = torch.sin
    model2.phi3 = torch.sin
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_tensor_type(torch.DoubleTensor)

    global a                                                          # domain (a,b) x (c,d)
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
    xy_inter = get_interior_point(N1)
    xy_inter = xy_inter.to(device)
    A = np.zeros((N1, 2*m))
    fv = np.zeros((N1, 1))
    laplaceu1 = -a1*(fduxx(model1, xy_inter, hx2) + fduyy(model1, xy_inter, hx2))
    laplaceu2 = -a2*(fduxx(model2, xy_inter, hx2) + fduyy(model2, xy_inter, hx2))
    for i in range(N1):
        if xy_inter[i:i+1, 0:1]**2 + xy_inter[i:i+1 ,1:2]**2 <= 0.75**2:
            A[i, 0:m] = laplaceu1[i:i+1, :].detach().cpu().numpy()
            fv[i, :] = f1(xy_inter[i:i+1,:]).cpu().numpy()
        else:
            A[i, m:2*m] = laplaceu2[i:i+1, :].detach().cpu().numpy()
            fv[i, :] = f2(xy_inter[i:i+1,:]).cpu().numpy()

    # interface couple
    xy_interface = get_interface_point(N3)[0]
    n1 = get_interface_point(N3)[1]
    n2 = get_interface_point(N3)[2]
    xy_interface = xy_interface.to(device)

    # [u]
    C = np.zeros((N3, 2*m))
    fc = np.zeros((N3, 1))
    C[:, 0:m] = (model1(xy_interface)[0]).detach().cpu().numpy()
    C[:, m:2*m] = -(model2(xy_interface)[0]).detach().cpu().numpy()
    fc[:, 0:1] = (u1(xy_interface) - u2(xy_interface)).cpu().numpy()

    # [\beta*gradu*n]
    D = np.zeros((N3, 2*m))
    fd = np.zeros((N3, 1))
    left_jump1 = a1*(fdux(model1,xy_interface,hx1)*n1 + fduy(model1,xy_interface,hx1)*n2)
    left_jump2 = -a2*(fdux(model2,xy_interface,hx1)*n1 + fduy(model2,xy_interface,hx1)*n2)
    right_jump1 = a1*(ux1(xy_interface)*n1 + uy1(xy_interface)*n2)
    right_jump2 = -a2*(ux2(xy_interface)*n1 + uy2(xy_interface)*n2)
    D[:,0:m] = left_jump1.detach().cpu().numpy()
    D[:, m:2*m] = left_jump2.detach().cpu().numpy()
    fd[:,0:1] = (right_jump1 + right_jump2).cpu().numpy()

    # boundary equations
    B = np.zeros((4*N2,2*m))
    xb = get_boundary_points0(N2)
    xb = xb.to(device)
    B[:, m:2*m] = (model2(xb)[0]).detach().cpu().numpy()
    fb = (u2(xb)).view([xb.size(0), 1]).cpu().numpy()

    # solving linear system
    matrix_a = np.vstack((A, p1*B, p2*C, p2*D))
    matrix_b = np.vstack((fv, p1*fb, p2*fc, p2*fd))
    wu = scipy.linalg.lstsq(matrix_a, matrix_b)[0]
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

    # endtime = datetime.datetime.now()
    # lasttime1 = (endtime - starttime).seconds
    # lasttime2 = (endtime - starttime).microseconds
    # print('Running time: %d.%ds' % (lasttime1, lasttime2))
    end = time.perf_counter()
    ttt = end - start
    print('Running time: %s Seconds' % (ttt))

    # calculation of L2-error
    xr = get_GL_points(n_int)
    wr = get_weights(n_int)
    wr = wr.reshape([n_int**2, 1])
    wr = wr.to(device)
    wr.requires_grad_()
    xr = xr.to(device)
    xr.requires_grad_()
    x = torch.zeros_like(xr)
    x[:, 0] = (b - a) / 2 * xr[:, 0] + (a + b) / 2
    x[:, 1] = (d - c) / 2 * xr[:, 1] + (c + d) / 2
    output_u = torch.zeros(n_int ** 2, 1)
    u_exact = torch.zeros(n_int ** 2, 1)
    for i in range(n_int ** 2):
        if x[i:i+1 , 0:1]**2 + x[i:i+1 , 1:2]**2 <= 0.75**2:
            output_u[i:i + 1, 0:1] = model1(x[i:i + 1, :])[1]
            u_exact[i:i + 1, 0:1] = u1(x[i:i + 1, :])
        else:
            output_u[i:i + 1, 0:1] = model2(x[i:i + 1, :])[1]
            u_exact[i:i + 1, 0:1] = u2(x[i:i + 1, :])
    err_L2 = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(torch.pow(output_u - u_exact, 2) * wr))
    u_L2 = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(torch.pow(u_exact, 2) * wr))
    err_rL2 = err_L2 / u_L2
    print('Final L^2 error:', err_L2.item())
    print('Final u_L^2 norm:', u_L2.item())
    print('Final L^2 relative error:', err_rL2.item())
    return err_rL2, ttt
    # return err_rL2


    # # pictures of numerical solution, real solution and error
    # with torch.no_grad():
    #     x1 = torch.linspace(a, b, 300)
    #     x2 = torch.linspace(c, d, 300)
    #     Y, X = torch.meshgrid(x2, x1)                                    # indexing='xy' or 'ij'
    #     Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    #     Z = Z.to(device)
    #     pred1 = torch.zeros(90000, 1)
    #     pred2 = torch.zeros(90000, 1)
    #     pred3 = torch.zeros(90000, 1)
    #     for i in range(90000):
    #         if Z[i:i+1, 0:1]** 2 + Z[i:i+1, 1:2]**2 <= 0.75**2:
    #             pred1[i, 0] = model1(Z[i:i+1,:])[1]
    #             pred2[i, 0] = u1(Z[i:i+1,:])
    #             pred3[i, 0] = torch.abs(model1(Z[i:i + 1, :])[1] - u1(Z[i:i+1,:]))
    #         else:
    #             pred1[i, 0] = model2(Z[i:i+1,:])[1]
    #             pred2[i, 0] = u2(Z[i:i + 1, :])
    #             pred3[i, 0] = torch.abs(model2(Z[i:i + 1, :])[1] - u2(Z[i:i + 1, :]))
    #
    # plt.figure()
    # pred1 = pred1.cpu().numpy()
    # pred1 = pred1.reshape(300, 300)
    # ax = plt.subplot(1,1,1)
    # h = plt.imshow(pred1, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    # plt.show()
    #
    # plt.figure()
    # pred2 = pred2.cpu().numpy()
    # pred2 = pred2.reshape(300, 300)
    # ax = plt.subplot(1,1,1)
    # h = plt.imshow(pred2, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    # plt.show()
    #
    # plt.figure()
    # pred3 = pred3.cpu().numpy()
    # pred3 = pred3.reshape(300, 300)
    # ax = plt.subplot(1,1,1)
    # h = plt.imshow(pred3, interpolation='nearest', cmap='rainbow',
    #                extent=[a, b, c, d],
    #                origin='lower', aspect=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(h, cax=cax)
    # plt.show()

# if __name__ == '__main__':
#     args = get_args()
#     device = torch.device('cpu')
#     main(r1=args.range1, hx1=args.difference1, hx2=args.difference2, N1=args.numbers1,
#          N2=args.numbers2, N3=args.numbers3, n_int=args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha1=args.parameter1,
#          alpha2=args.parameter2, device=device)

# # change parameters
# if __name__ == '__main__':
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     data = np.zeros((40, 40))
#     args = get_args()
#     device = torch.device('cpu')
#     for i in range(40):
#         for j in range(40):
#             r22 = 0.2*i + 0.2
#             r23 = 0.2*j + 0.2
#             data[i,j] = main(r22, r23, hx1=args.difference1, hx2=args.difference2, N1=args.numbers1,
#               N2=args.numbers2, N3=args.numbers3, n_int=args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha1=args.parameter1,
#               alpha2=args.parameter2, device=device)
#     print(data)
#     data_df = pd.DataFrame(data)  # Convert the ndarray to a DataFrame
#
#     with pd.ExcelWriter('deep_test_tanh_tanh_tanh_2.xlsx', engine='openpyxl') as writer:
#         data_df.to_excel(writer, sheet_name='Sheet1')


# run ten times and give the average value
if __name__ == '__main__':
    args = get_args()
    device = torch.device('cpu')
    data = np.zeros((11,1))
    ti = np.zeros((11,1))
    aaa = 0
    bbb = 0
    for i in range(10):
        data[i,0],ti[i,0] = main(r1=args.range1, hx1=args.difference1, hx2=args.difference2, N1=args.numbers1,
             N2=args.numbers2, N3=args.numbers3, n_int=args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha1=args.parameter1,
             alpha2=args.parameter2, device=device)
        aaa += data[i,0]
        bbb += ti[i,0]
    data[10,0] = aaa/10
    ti[10,0] = bbb/10
    print('%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e ' % (data[0,0],data[1,0],data[2,0],data[3,0],data[4,0],data[5,0],data[6,0],data[7,0],data[8,0],data[9,0],data[10,0]))
    print('\n')
    print('%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e ' % (ti[0,0],ti[1,0],ti[2,0],ti[3,0],ti[4,0],ti[5,0],ti[6,0],ti[7,0],ti[8,0],ti[9,0],ti[10,0]))




