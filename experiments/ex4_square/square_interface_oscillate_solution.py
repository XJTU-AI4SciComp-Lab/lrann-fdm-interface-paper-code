"""LRaNN-FDM paper experiment; see docs/reproduction.md for settings."""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
import lrann_interface.compat

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

class dnn(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)  # output layer
        self.phi = torch.tanh

    def forward(self, x):
        x = self.phi(self.hidden(x))  # activation function for hidden layer
        y = self.predict(x)  # linear output
        return x,y

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

from lrann_interface.finite_difference import fduxx as fduxx

from lrann_interface.finite_difference import fduyy as fduyy

from lrann_interface.finite_difference import fdux_variant2 as fdux

from lrann_interface.finite_difference import fduy_variant2 as fduy


from lrann_interface.quadrature import gauss_points_2d as get_GL_points

from lrann_interface.quadrature import gauss_weights_2d as get_weights

def get_boundary_points0(N):
    index1 = (b-a)*torch.rand(N, 1) + a
    index2 = (d-c)*torch.rand(N, 1) + c
    xb1 = torch.cat((index1, c*torch.ones_like(index2)), dim=1)
    xb2 = torch.cat((index1, d*torch.ones_like(index2)), dim=1)
    xb3 = torch.cat((a * torch.ones_like(index1),index2), dim=1)
    xb4 = torch.cat((b * torch.ones_like(index1),index2), dim=1)
    xb = torch.cat((xb1,xb2,xb3,xb4), dim=0)
    return xb


def get_interior_point(N):
    index1 = (b-a)*torch.rand(N, 1) + a
    index2 = (d-c)*torch.rand(N, 1) + c
    xb = torch.cat((index1, index2), dim=1)
    return xb


def get_interface_point(N):
    index1 = (0.5*(b-a))*torch.rand(N, 1) + 0.5*(a+0.5*(a+b))
    index2 = (0.5*(d-c))*torch.rand(N, 1) + 0.5*(c+0.5*(c+d))
    xb1 = torch.cat((index1, 0.5*(c+0.5*(c+d)) * torch.ones_like(index2)), dim=1)
    xb2 = torch.cat((index1, 0.5*(d+0.5*(c+d)) * torch.ones_like(index2)), dim=1)
    xb3 = torch.cat((0.5*(a+0.5*(a+b)) * torch.ones_like(index1), index2), dim=1)
    xb4 = torch.cat((0.5*(b+0.5*(a+b)) * torch.ones_like(index1), index2), dim=1)
    xb = torch.cat((xb1, xb2, xb3, xb4), dim=0)
    return xb


def u1(x, k):
    ini = 0
    for i in range(k):
        ini = ini + torch.sin(2**(i+1)*math.pi*x[:, 0:1])*torch.sin(2**(i+1)*math.pi*x[:, 1:2])
    return 1/k * ini


def u2(x, k):
    ini = 0
    for i in range(k):
        ini = ini + torch.cos(2**(i+1)*math.pi*x[:, 0:1])*torch.cos(2**(i+1)*math.pi*x[:, 1:2])
    return 1/k * ini


def u1x(x, k):
    ini = 0
    for i in range(k):
        ini = ini + 2**(i+1)*math.pi*torch.cos(2**(i+1)*math.pi*x[:, 0:1])*torch.sin(2**(i+1)*math.pi*x[:, 1:2])
    return 1/k * ini


def u1y(x, k):
    ini = 0
    for i in range(k):
        ini = ini + 2**(i+1)*math.pi*torch.sin(2**(i+1)*math.pi*x[:, 0:1])*torch.cos(2**(i+1)*math.pi*x[:, 1:2])
    return 1/k * ini


def u2x(x, k):
    ini = 0
    for i in range(k):
        ini = ini - 2**(i+1)*math.pi*torch.sin(2**(i+1)*math.pi*x[:, 0:1])*torch.cos(2**(i+1)*math.pi*x[:, 1:2])
    return 1/k * ini


def u2y(x, k):
    ini = 0
    for i in range(k):
        ini = ini - 2**(i+1)*math.pi*torch.cos(2**(i+1)*math.pi*x[:, 0:1])*torch.sin(2**(i+1)*math.pi*x[:, 1:2])
    return 1/k * ini


def f1(x, k):
    ini = 0
    for i in range(k):
        ini = ini - 2**(2*i+3)*math.pi**2*torch.sin(2**(i+1)*math.pi*x[:,0:1])*torch.sin(2**(i+1)*math.pi*x[:,1:2])
    return -a1 * 1/k * ini


def f2(x, k):
    ini = 0
    for i in range(k):
        ini = ini - 2**(2*i+3)*math.pi**2*torch.cos(2**(i+1)*math.pi*x[:,0:1])*torch.cos(2**(i+1)*math.pi*x[:,1:2])
    return -a2 * 1/k * ini


#(1,10):1.8,0.6;
def get_args():  # Parameters
    parser = argparse.ArgumentParser(description='Train the pde net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r1', '--range1',  default=4.1, help='Range of initial weights', type=float)
    parser.add_argument('-r2', '--range2', default=2, help='Range of initial weights', type=float)
    parser.add_argument('-hx1', '--difference1', default=1e-6, help='difference parameter', type=float)
    parser.add_argument('-hx2', '--difference2', default=5e-4, help='difference parameter', type=float)
    parser.add_argument('-n1', '--numbers1', default=3600, help='numbers of the interior points', type=int)
    parser.add_argument('-n2', '--numbers2', default=60, help='number of the boundary points', type=int)
    parser.add_argument('-n3', '--numbers3', default=240, help='number of the interface points', type=int)
    parser.add_argument('-n_int', '--numbers4', default=20, help='number of the integration points', type=int)
    parser.add_argument('-m', '--neurons', default=640, help='number of hidden layer nodes', type=int)
    parser.add_argument('-p1', '--penalty1', default=100, help='penalty parameter on boundary', type=float)
    parser.add_argument('-p2', '--penalty2', default=100, help='penalty parameter on interface', type=float)
    parser.add_argument('-alpha1', '--parameter1', default=1, help='coefficient1 of solution', type=float)
    parser.add_argument('-alpha2', '--parameter2', default=1, help='coefficient2 of solution', type=float)
    parser.add_argument('-k1', '--parameter3', default=3, help='coefficient3 of solution', type=float)
    parser.add_argument('-k2', '--parameter4', default=2, help='coefficient4 of solution', type=float)
    return parser.parse_args()


def main(r1,r2,hx1,hx2,N1,N2,N3,n_int,m,p1,p2,alpha1,alpha2,k1,k2,device): # Solve for the output weights

    # Set network parameters
    global ra1
    global rb1
    global ra2
    global rb2
    ra1 = -r1
    rb1 = r1
    ra2 = -r2
    rb2 = r2
    in_N = 2
    out_N = 1
    model1 = dnn(in_N, m, out_N).to(device)
    model1 = model1.double()
    model1.apply(weights_init1)

    model2 = dnn(in_N, m, out_N).to(device)
    model2 = model2.double()
    model2.apply(weights_init2)
    torch.set_default_tensor_type(torch.DoubleTensor)


    global a  # (a,b) x (c,d)
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

    starttime = time.perf_counter()


    #Interior sampling
    xy_inter = get_interior_point(N1)
    xy_inter = xy_inter.to(device)

    A = np.zeros((N1,2*m))
    fv = np.zeros((N1,1))
    laplace1 = -a1*(fduxx(model1, xy_inter, hx2) + fduyy(model1, xy_inter, hx2))
    laplace2 = -a2*(fduxx(model2, xy_inter, hx2) + fduyy(model2, xy_inter, hx2))
    for i in range(N1):
        if 0.5*(a+0.5*(a+b)) <= xy_inter[i:i+1,0] <= 0.5*(b+0.5*(a+b)) and 0.5*(c+0.5*(c+d)) <= xy_inter[i:i+1,1] <= 0.5*(d+0.5*(c+d)):
            A[i,0:m] = (laplace1[i:i+1, :]).detach().cpu().numpy()
            fv[i,:] = f1(xy_inter[i:i+1,:], k1).cpu().numpy()
        else:
            A[i,m:2*m] = (laplace2[i:i+1, :]).detach().cpu().numpy()
            fv[i,:] = f2(xy_inter[i:i+1,:], k2).cpu().numpy()

    xy_interface = get_interface_point(N3)
    xy_interface = xy_interface.to(device)

    n1 = torch.zeros(4*N3,1)
    n2 = torch.zeros(4*N3,1)
    n1[2*N3:3*N3,0:1] = -1
    n1[3*N3:4*N3,0:1] = 1
    n2[0:N3,0:1] = -1
    n2[N3:2*N3,0:1] = 1
    n = torch.cat((n1,n2), dim=1)

    C = np.zeros((4*N3, 2*m))
    fc = np.zeros((4*N3, 1))
    uh_jump1 = model1(xy_interface)[0]
    uh_jump2 = -model2(xy_interface)[0]
    C[:,0:m] = (uh_jump1).detach().cpu().numpy()
    C[:,m:2*m] = (uh_jump2).detach().cpu().numpy()
    fc[:,0:1] = (u1(xy_interface, k1) - u2(xy_interface, k2)).cpu().numpy()

    D = np.zeros((4*N3,2*m))
    fd = np.zeros((4*N3,1))
    left_jump1 = a1*(fdux(model1,xy_interface,hx1)*n1 + fduy(model1,xy_interface,hx1)*n2)
    left_jump2 = -a2*(fdux(model2,xy_interface,hx1)*n1 + fduy(model2,xy_interface,hx1)*n2)
    right_jump1 = a1*(u1x(xy_interface,k1)*n1 + u1y(xy_interface,k1)*n2)
    right_jump2 = - a2*(u2x(xy_interface,k2)*n1 + u2y(xy_interface,k2)*n2)
    D[:, 0:m] = left_jump1.detach().cpu().numpy()
    D[:, m:2*m] = left_jump2.detach().cpu().numpy()
    fd[:, 0:1] = (right_jump1 + right_jump2).cpu().numpy()

    B = np.zeros((4*N2,2*m))
    xb = get_boundary_points0(N2)
    xb = xb.to(device)
    B[:,m:2*m] = (model2(xb)[0]).detach().cpu().numpy()
    fb = (u2(xb, k2)).view([xb.size(0), 1]).cpu().numpy()

    matrix_a = np.vstack((A, p1*B, p2*C, p2*D))  # Assemble the linear system
    matrix_b = np.vstack((fv, p1*fb, p2*fc, p2*fd))
    wu = np.linalg.lstsq(matrix_a, matrix_b)[0]  # Solve the least-squares system
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


    endtime = time.perf_counter()
    ttt = endtime - starttime
    print('Running time: %s Seconds' % (ttt))

    xr = get_GL_points(n_int)
    wr = get_weights(n_int)
    wr = wr.reshape([n_int**2,1])
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
        if 0.5*(a+0.5*(a+b)) <= x[i:i+1,0] <= 0.5*(b+0.5*(a+b)) and 0.5*(c+0.5*(c+d)) <= x[i:i+1,1] <= 0.5*(d+0.5*(c+d)):
            output_u[i:i + 1, 0:1] = model1(x[i:i + 1, :])[1]
            u_exact[i:i + 1, 0:1] = u1(x[i:i + 1, :], k1)
        else:
            output_u[i:i + 1, 0:1] = model2(x[i:i + 1, :])[1]
            u_exact[i:i + 1, 0:1] = u2(x[i:i + 1, :], k2)
    err_L2 = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(torch.pow(output_u - u_exact, 2) * wr))
    u_L2 = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(torch.pow(u_exact, 2) * wr))
    err_rL2 = err_L2 / u_L2
    print('Final L^2 error:', err_L2.item())
    print('Final u_L^2 norm:', u_L2.item())
    print('Final L^2 relative error:', err_rL2.item())
    return err_rL2, ttt


    # # Plot the solution and error
    # with torch.no_grad():
    #     x1 = torch.linspace(a, b, 300)
    #     x2 = torch.linspace(c, d, 300)
    #     Y, X = torch.meshgrid(x2, x1)  # indexing='xy' or 'ij'  Controls matrix transposition and image orientation
    #     Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    #     Z = Z.to(device)
    #     pred1 = torch.zeros(90000, 1)
    #     pred2 = torch.zeros(90000, 1)
    #     pred3 = torch.zeros(90000, 1)
    #     for i in range(90000):
    #         if 0.5*(a+0.5*(a+b)) <= Z[i:i+1,0] <= 0.5*(b+0.5*(a+b)) and 0.5*(c+0.5*(c+d)) <= Z[i:i+1,1] <= 0.5*(d+0.5*(c+d)):
    #             pred1[i, 0] = model1(Z[i:i+1,:])[1]
    #             pred2[i, 0] = u1(Z[i:i+1,:], k1)
    #             pred3[i, 0] = torch.abs(model1(Z[i:i + 1, :])[1] - u1(Z[i:i+1,:], k1))
    #         else:
    #             pred1[i, 0] = model2(Z[i:i+1,:])[1]
    #             pred2[i, 0] = u2(Z[i:i+1, :], k2)
    #             pred3[i, 0] = torch.abs(model2(Z[i:i + 1, :])[1] - u2(Z[i:i + 1, :], k2))
    #
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
    #
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
    #
    #
    #
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


# if __name__ == '__main__':
#     args = get_args()  # Change parameters
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
#     main(r1=args.range1, r2=args.range2, hx1=args.difference1, hx2=args.difference2, N1=args.numbers1,
#         N2=args.numbers2, N3=args.numbers3, n_int=args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha1=args.parameter1,
#         alpha2=args.parameter2, k1=args.parameter3, k2=args.parameter4, device=device)


if __name__ == '__main__':
    args = get_args()  # Change parameters
    device = torch.device('cpu')
    data = np.zeros((11,1))
    ti = np.zeros((11,1))
    aaa = 0
    bbb = 0
    for i in range(10):
        data[i,0],ti[i,0] = main(r1=args.range1, r2=args.range2, hx1=args.difference1, hx2=args.difference2, N1=args.numbers1,
        N2=args.numbers2, N3=args.numbers3, n_int=args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha1=args.parameter1,
        alpha2=args.parameter2, k1=args.parameter3, k2=args.parameter4, device=device)
        aaa += data[i, 0]
        bbb += ti[i,0]
    data[10, 0] = aaa / 10
    ti[10,0] = bbb/10
    print('%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e ' % (data[0,0],data[1,0],data[2,0],data[3,0],data[4,0],data[5,0],data[6,0],data[7,0],data[8,0],data[9,0],data[10,0]))
    print('\n')
    print('%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e ' % (ti[0,0],ti[1,0],ti[2,0],ti[3,0],ti[4,0],ti[5,0],ti[6,0],ti[7,0],ti[8,0],ti[9,0],ti[10,0]))



