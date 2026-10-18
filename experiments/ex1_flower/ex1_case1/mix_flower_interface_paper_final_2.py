"""LRaNN-FDM paper experiment; see docs/reproduction.md for settings."""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / 'src'))
import lrann_interface.compat

# Solve Two-dimensional Flower Shape Interface Problem (example4.1)
# mixed formulation

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
class dnn1(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)                             # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)                            # output layer
        self.phi = torch.tanh                                                                     # activation function

    def forward(self, x):
        x = self.phi(self.hidden(x))                                                              # output of hidden layer
        y = self.predict(x)                                                                       # linear output
        return x,y


class dnn2(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)
        self.phi = torch.tanh # Activation function

    def forward(self, x):
        x = self.phi(self.hidden(x))
        y = self.predict(x)
        y1 = y[:,0:1]
        y2 = y[:,1:2]
        return x,y1,y2


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


def weights_init3(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight,ra3,rb3)
        if m.bias is not None:
            nn.init.uniform_(m.bias,ra3,rb3)


def weights_init4(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight,ra4,rb4)
        if m.bias is not None:
            nn.init.uniform_(m.bias,ra4,rb4)


# approximating derivative by finite difference
# model: function, xr: sampling points, h: difference step
# first-derivative with respect to x
from lrann_interface.finite_difference import fdux_variant2 as fdux


# first-derivative with respect to y
from lrann_interface.finite_difference import fduy_variant2 as fduy


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
def get_interface_point(N):
    theta = 2*np.pi*torch.rand(N,1)
    index1 = 0.02*math.sqrt(5)+(0.4+0.2*torch.sin(20*theta))*torch.cos(theta)
    index2 = 0.02*math.sqrt(5)+(0.4+0.2*torch.sin(20*theta))*torch.sin(theta)
    xb = torch.cat((index1, index2), dim=1)
    return xb,theta


# real solution
# solution in Omega1
def u1(x):
    return 1/a1*torch.exp(x[:,0:1]*x[:,1:2])


# solution in Omega2
def u2(x):
    return 1/a2*torch.mul(torch.sin(np.pi*x[:,0:1]),torch.sin(np.pi*x[:,1:2]))


# partial derivative of real solution
def ux1(x):
    return 1/a1*x[:,1:2]*torch.exp(x[:,0:1]*x[:,1:2])


def ux2(x):
    return 1/a2*np.pi*torch.mul(torch.cos(np.pi*x[:,0:1]),torch.sin(np.pi*x[:,1:2]))


def uy1(x):
    return 1/a1*x[:,0:1]*torch.exp(x[:,0:1]*x[:,1:2])


def uy2(x):
    return 1/a2*np.pi*torch.mul(torch.sin(np.pi*x[:,0:1]),torch.cos(np.pi*x[:,1:2]))


# source term
def f1(x):
    return -(x[:,0:1]**2+x[:,1:2]**2)*torch.exp(x[:,0:1]*x[:,1:2])


def f2(x):
    return 2*np.pi**2*torch.mul(torch.sin(np.pi*x[:,0:1]),torch.sin(np.pi*x[:,1:2]))


# parameters
def get_args():
    parser = argparse.ArgumentParser(description='Train the pde net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r1', '--range1', default=1, help='Range of initial weights', type=float)
    parser.add_argument('-r2', '--range2', default=1, help='Range of initial weights', type=float)
    parser.add_argument('-r3', '--range3', default=1, help='Range of initial weights', type=float)
    parser.add_argument('-r4', '--range4', default=1, help='Range of initial weights', type=float)
    parser.add_argument('-hx1', '--difference1', default=1e-6, help='difference parameter', type=float)
    parser.add_argument('-n1', '--numbers1', default=400, help='numbers of the interior points', type=int)
    parser.add_argument('-n2', '--numbers2', default=20, help='number of the boundary points', type=int)
    parser.add_argument('-n3', '--numbers3', default=80, help='number of the interface points', type=int)
    parser.add_argument('-n_int', '--numbers4', default=20, help='number of the integration points', type=int)
    parser.add_argument('-m', '--neurons', default=640, help='number of hidden layer nodes', type=int)
    parser.add_argument('-p1', '--penalty1', default=100, help='penalty parameter on boundary', type=float)
    parser.add_argument('-p2', '--penalty2', default=100, help='penalty parameter on interface', type=float)
    parser.add_argument('-alpha1', '--parameter1', default=1, help='coefficient1 of solution', type=float)
    parser.add_argument('-alpha2', '--parameter2', default=10, help='coefficient2 of solution', type=float)
    return parser.parse_args()


def main(r1, r2, r3, r4, hx1, N1, N2, N3, n_int, m, p1, p2, alpha1, alpha2, device): # Solve for the output weights
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
    out_N2 = 2

    model1 = dnn1(in_N, m, out_N).to(device)                          # approximate real solution in Omega1
    model1 = model1.double()
    model1.apply(weights_init1)

    model2 = dnn1(in_N, m, out_N).to(device)                          # approximate real solution in Omega2
    model2 = model2.double()
    model2.apply(weights_init2)

    model3 = dnn2(in_N, m, out_N2).to(device)                         # approximate p1 = a1*gradu1 in Omega1
    model3 = model3.double()
    model3.apply(weights_init3)

    model4 = dnn2(in_N, m, out_N2).to(device)                         # approximate p2 = a2*gradu2 in Omega2
    model4 = model4.double()
    model4.apply(weights_init4)

    torch.set_default_tensor_type(torch.DoubleTensor)

# Set parameters

    global a                                                          # domain(a,b) x (c,d)
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
    A1 = np.zeros((N1,6*m))
    A2 = np.zeros((N1,6*m))
    A3 = np.zeros((N1,6*m))
    fv1 = np.zeros((N1,1))
    fv2 = np.zeros((N1,1))
    fv3 = np.zeros((N1,1))
    test = (xy_inter[:,0:1] - 0.02*math.sqrt(5))**2 + (xy_inter[:,1:2] - 0.02*math.sqrt(5))**2
    theta_1 = torch.atan2((xy_inter[:, 1:2] - 0.02 * math.sqrt(5)), (xy_inter[:, 0:1] - 0.02 * math.sqrt(5)))
    for i in range(N1):
        if test[i] <= (0.4+0.2*torch.sin(20*theta_1[i]))**2:
            A1[i,2*m:3*m] = -a1*(fdux(model3, xy_inter[i:i+1], hx1)).detach().cpu().numpy()
            A1[i,3*m:4*m] = -a1*(fduy(model3, xy_inter[i:i+1], hx1)).detach().cpu().numpy()
            A2[i,2*m:3*m] = model3(xy_inter[i:i+1])[0].detach().cpu().numpy()
            A2[i,0:m] = -fdux(model1, xy_inter[i:i+1], hx1).detach().cpu().numpy()
            A3[i,3*m:4*m] = model3(xy_inter[i:i+1])[0].detach().cpu().numpy()
            A3[i,0:m] = -fduy(model1, xy_inter[i:i + 1], hx1).detach().cpu().numpy()
            fv1[i,0:1] = f1(xy_inter[i:i+1,:]).cpu().numpy()
        else:
            A1[i,4*m:5*m] = -a2*fdux(model4, xy_inter[i:i+1], hx1).detach().cpu().numpy()
            A1[i,5*m:6*m] = -a2*fduy(model4, xy_inter[i:i+1], hx1).detach().cpu().numpy()
            A2[i,4*m:5*m] = model4(xy_inter[i:i+1])[0].detach().cpu().numpy()
            A2[i,m:2*m] = -fdux(model2, xy_inter[i:i+1], hx1).detach().cpu().numpy()
            A3[i,5*m:6*m] = model4(xy_inter[i:i+1])[0].detach().cpu().numpy()
            A3[i,m:2*m] = -fduy(model2, xy_inter[i:i + 1], hx1).detach().cpu().numpy()
            fv1[i,0:1] = f2(xy_inter[i:i+1,:]).cpu().numpy()

    # interface couple
    xy_interface = get_interface_point(N3)[0]
    theta_interface = get_interface_point(N3)[1]
    xy_interface = xy_interface.to(device)

    # calculation of normal vector
    rr = 0.4 + 0.2*torch.sin(20*theta_interface)
    rrd = 4*torch.cos(20*theta_interface)
    h = torch.pow(rr**2 + rrd**2,-1/2)
    n1 = -h*(rrd*torch.sin(theta_interface)+rr*torch.cos(theta_interface))
    n2 = h*(rrd*torch.cos(theta_interface)-rr*torch.sin(theta_interface))

    C = np.zeros((N3,6*m))
    fc = np.zeros((N3,1))
    C[:,0:m] = model1(xy_interface)[0].detach().cpu().numpy()
    C[:,m:2*m] = -model2(xy_interface)[0].detach().cpu().numpy()
    fc[:,0:1] = (u1(xy_interface) - u2(xy_interface)).cpu().numpy()

    D = np.zeros((N3,6*m))
    fd = np.zeros((N3,1))
    left_jump11 = a1*model3(xy_interface)[0]*n1
    left_jump12 = a1*model3(xy_interface)[0]*n2
    left_jump21 = -a2*model4(xy_interface)[0]*n1
    left_jump22 = -a2*model4(xy_interface)[0]*n2
    right_jump1 = a1*(ux1(xy_interface)*n1 + uy1(xy_interface)*n2)
    right_jump2 = - a2*(ux2(xy_interface)*n1 + uy2(xy_interface)*n2)
    D[:, 2 * m:3 * m] = left_jump11.detach().cpu().numpy()
    D[:, 3 * m:4 * m] = left_jump12.detach().cpu().numpy()
    D[:, 4 * m:5 * m] = left_jump21.detach().cpu().numpy()
    D[:, 5 * m:6 * m] = left_jump22.detach().cpu().numpy()
    fd[:,0:1] = (right_jump1 + right_jump2).cpu().numpy()

    # boundary equations
    B = np.zeros((4*N2,6*m))
    xb = get_boundary_points0(N2)
    xb = xb.to(device)
    B[:,m:2*m] = (model2(xb)[0]).detach().cpu().numpy()
    fb = (u2(xb)).view([xb.size(0), 1]).cpu().numpy()

    # solving linear system
    matrix_a = np.vstack((A1, A2, A3, p1*B, p2*C, p2*D))
    matrix_b = np.vstack((fv1, fv2, fv3, p1*fb, p2*fc, p2*fd))
    wu = np.linalg.lstsq(matrix_a, matrix_b)[0]
    wu1 = torch.transpose((torch.from_numpy(wu[0:m]).to(device)), 0, 1)
    wu2 = torch.transpose((torch.from_numpy(wu[m:2 * m]).to(device)), 0, 1)
    wu3 = torch.transpose((torch.from_numpy(wu[2 * m:3 * m]).to(device)), 0, 1)
    wu4 = torch.transpose((torch.from_numpy(wu[3 * m:4 * m]).to(device)), 0, 1)
    wu5 = torch.transpose((torch.from_numpy(wu[4 * m:5 * m]).to(device)), 0, 1)
    wu6 = torch.transpose((torch.from_numpy(wu[5 * m:6 * m]).to(device)), 0, 1)
    wuu1 = wu1
    wuu2 = wu2
    wup1 = torch.cat((wu3, wu4), dim=0)
    wup2 = torch.cat((wu5, wu6), dim=0)

    for name, para in model1.named_parameters():
        if 'predict.weight' in name:
            para.data = wuu1
    for name, para in model2.named_parameters():
        if 'predict.weight' in name:
            para.data = wuu2
    for name, para in model3.named_parameters():
        if 'predict.weight' in name:
            para.data = wup1
    for name, para in model4.named_parameters():
        if 'predict.weight' in name:
            para.data = wup2

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
    center = torch.ones(n_int ** 2, 1) * 0.02 * math.sqrt(5)
    test2 = (x[:, 0:1] - center) ** 2 + (x[:, 1:2] - center) ** 2
    theta_2 = torch.atan2((x[:, 1:2] - 0.02 * math.sqrt(5)), (x[:, 0:1] - 0.02 * math.sqrt(5)))
    output_u = torch.zeros(n_int ** 2, 1)
    u_exact = torch.zeros(n_int ** 2, 1)
    output_p1 = torch.zeros(n_int ** 2, 1)
    output_p2 = torch.zeros(n_int ** 2, 1)
    p1_exact = torch.zeros(n_int ** 2, 1)
    p2_exact = torch.zeros(n_int ** 2, 1)
    for i in range(n_int ** 2):
        if test2[i] <= (0.4 + 0.2 * torch.sin(20 * theta_2[i])) ** 2:
            output_u[i:i + 1, 0:1] = model1(x[i:i + 1, :])[1]
            u_exact[i:i + 1, 0:1] = u1(x[i:i + 1, :])
            output_p1[i:i+1,0:1] = model3(x[i:i+1,:])[1]
            output_p2[i:i + 1, 0:1] = model3(x[i:i + 1, :])[2]
            p1_exact[i:i + 1, 0:1] = ux1(x[i:i + 1, :])
            p2_exact[i:i + 1, 0:1] = uy1(x[i:i + 1, :])
        else:
            output_u[i:i + 1, 0:1] = model2(x[i:i + 1, :])[1]
            u_exact[i:i + 1, 0:1] = u2(x[i:i + 1, :])
            output_p1[i:i+1,0:1] = model4(x[i:i+1,:])[1]
            output_p2[i:i + 1, 0:1] = model4(x[i:i + 1, :])[2]
            p1_exact[i:i + 1, 0:1] = ux2(x[i:i + 1, :])
            p2_exact[i:i + 1, 0:1] = uy2(x[i:i + 1, :])
    err_L2 = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(torch.pow(output_u - u_exact, 2) * wr))
    err_L2_p = torch.sqrt((b - a) * (d - c) / 4 * torch.sum((torch.pow(output_p1 - p1_exact, 2)+torch.pow(output_p2 - p2_exact, 2)) * wr))
    u_L2 = torch.sqrt((b - a) * (d - c) / 4 * torch.sum(torch.pow(u_exact, 2) * wr))
    p_L2 = torch.sqrt((b - a) * (d - c) / 4 * torch.sum((torch.pow(p1_exact, 2)+torch.pow(p2_exact, 2)) * wr))
    err_rL2 = err_L2 / u_L2.detach()
    err_rL2_p = err_L2_p / p_L2.detach()
    print('Final L^2 error:', err_L2.item())
    print('Final u_L^2 norm:', u_L2.item())
    print('Final L^2 relative error:', err_rL2.item())
    print('Final p_L^2 error:', err_L2_p.item())
    print('Final p_u_L^2 norm:', p_L2.item())
    print('Final p_L^2 relative error:', err_rL2_p.item())
    results = torch.zeros(1, 2)
    results[0, 0] = err_rL2
    results[0, 1] = err_rL2_p
    return results

    # # pictures of numerical solution, real solution and error
    # with torch.no_grad():
    #     x1 = torch.linspace(a, b, 300)
    #     x2 = torch.linspace(c, d, 300)
    #     Y, X = torch.meshgrid(x1, x2)  # indexing='xy' or 'ij'
    #     Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    #     Z = Z.to(device)
    #     pred1 = torch.zeros(90000, 1)
    #     pred2 = torch.zeros(90000, 1)
    #     pred3 = torch.zeros(90000, 1)
    #     center = torch.ones(90000, 1) * 0.02*math.sqrt(5)
    #     test3 = (Z[:,0:1] - center) ** 2 + (Z[:,1:2] - center) ** 2
    #     theta_3 = torch.atan2((Z[:, 1:2] - 0.02 * math.sqrt(5)), (Z[:, 0:1] - 0.02 * math.sqrt(5)))
    #     for i in range(90000):
    #         if test3[i] <= (0.4+0.2*torch.sin(20*theta_3[i]))**2:
    #             pred1[i, 0] = model1(Z[i:i+1,:])[1]
    #             pred2[i, 0] = u1(Z[i:i+1,:])
    #             pred3[i, 0] = torch.abs(model1(Z[i:i + 1, :])[1] - u1(Z[i:i+1,:]))
    #         else:
    #             pred1[i, 0] = model2(Z[i:i+1,:])[1]
    #             pred2[i, 0] = u2(Z[i:i + 1, :])
    #             pred3[i, 0] = torch.abs(model2(Z[i:i + 1, :])[1] - u2(Z[i:i + 1, :]))
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
#     args = get_args()
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
#     main(r1=args.range1, r2=args.range2, r3=args.range3, r4=args.range4, hx1=args.difference1, N1=args.numbers1,
#          N2=args.numbers2, N3=args.numbers3, n_int=args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha1=args.parameter1,
#          alpha2=args.parameter2, device=device)


# run ten times and give the average value
if __name__ == '__main__':
    args = get_args()  # Change parameters
    device = torch.device('cpu')
    data = torch.zeros(11,2)
    aaa = 0
    bbb = 0
    for i in range(10):
        data[i:i+1,:] = main(r1=args.range1, r2=args.range2,r3=args.range3,r4=args.range4, hx1=args.difference1, N1=args.numbers1,
             N2=args.numbers2, N3=args.numbers3, n_int=args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha1=args.parameter1,
             alpha2=args.parameter2, device=device)
        aaa += data[i,0]
        bbb += data[i,1]
    data[10,0] = aaa/10
    data[10,1] = bbb/10
    print('%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e ' % (data[0,0],data[1,0],data[2,0],data[3,0],data[4,0],data[5,0],data[6,0],data[7,0],data[8,0],data[9,0],data[10,0]))
    print('\n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e ' % (data[0,1],data[1,1],data[2,1],data[3,1],data[4,1],data[5,1],data[6,1],data[7,1],data[8,1],data[9,1],data[10,1]))



