"""LRaNN-FDM paper experiment; see docs/reproduction.md for settings."""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
import lrann_interface.compat

# Solve Three-dimensional Spherical Interface Problem (example4.2)
# mixed formulation

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.polynomial.legendre import leggauss
from itertools import product
import argparse
import datetime
import math


# define FNN
class dnn1(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)              # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)             # output layer
        self.phi = torch.tanh                                                      # activation function

    def forward(self, x):
        x = self.phi(self.hidden(x))                                               # output of hidden layer
        y = self.predict(x)                                                        # linear output
        return x,y


class dnn2(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)
        self.phi = torch.tanh

    def forward(self, x):
        x = self.phi(self.hidden(x))
        y = self.predict(x)
        y1 = y[:,0:1]
        y2 = y[:,1:2]
        y3 = y[:,2:3]
        return x,y1,y2,y3


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
from lrann_interface.finite_difference import fdux as fdux


from lrann_interface.finite_difference import fduy as fduy


def fduz(model,xr,h):
    ei = torch.zeros_like(xr)
    ei[:,2]=1
    uz = (model(xr+h*ei)[0] - model(xr-h*ei)[0])/(2*h)
    return uz


# Gauss-Legendre numerical integration(3-D)
# integration points
from lrann_interface.quadrature import gauss_points_3d as get_GL_points


from lrann_interface.quadrature import gauss_weights_3d as get_weights


# sampling points
# boundary points
def get_boundary_points0(N):
    index1 = (b - a) * torch.rand(N, 1) + a
    index2 = (d - c) * torch.rand(N, 1) + c
    index3 = (f - e) * torch.rand(N, 1) + e
    xb1 = torch.cat((index1, index2, e * torch.ones_like(index1)), dim=1)
    xb2 = torch.cat((index1, index2, f * torch.ones_like(index1)), dim=1)
    xb3 = torch.cat((index1, c * torch.ones_like(index2), index3), dim=1)
    xb4 = torch.cat((index1, d * torch.ones_like(index2), index3), dim=1)
    xb5 = torch.cat((a * torch.ones_like(index3), index2, index3), dim=1)
    xb6 = torch.cat((b * torch.ones_like(index3), index2, index3), dim=1)
    xb = torch.cat((xb1, xb2, xb3, xb4, xb5, xb6), dim=0)
    return xb


# interior points
def get_interior_point(N):
    index1 = (b-a)*torch.rand(N, 1) + a
    index2 = (d-c)*torch.rand(N, 1) + c
    index3 = (f-e)*torch.rand(N, 1) + e
    xb = torch.cat((index1, index2, index3), dim=1)
    return xb


# interface points
def get_interface_point(N):
    theta1 = 2*np.pi*torch.rand(N,1)
    theta2 = 2*np.pi*torch.rand(N,1)
    index1 = 0.75 * torch.cos(theta1) * torch.sin(theta2)
    index2 = 0.75 * torch.cos(theta1) * torch.cos(theta2)
    index3 = 0.75 * torch.sin(theta1)
    xb = torch.cat((index1, index2, index3), dim=1)
    n1 = torch.cos(theta1) * torch.sin(theta2)
    n2 = torch.cos(theta1) * torch.cos(theta2)
    n3 = torch.sin(theta1)
    return xb,n1,n2,n3


# real solution
# solution in Omega1
def u1(x):
    return 5*torch.exp(x[:,0:1]**2+x[:,1:2]**2+x[:,2:3]**2)+20


# solution in Omega2
def u2(x):
    return 10*(x[:,0:1]+x[:,1:2]+x[:,2:3])


# partial derivative of real solution
def ux1(x):
    return 10*x[:,0:1]*torch.exp(x[:,0:1]**2+x[:,1:2]**2+x[:,2:3]**2)


def ux2(x):
    return 10*torch.ones(x.shape[0],1)

def uy1(x):
    return 10*x[:,1:2]*torch.exp(x[:,0:1]**2+x[:,1:2]**2+x[:,2:3]**2)


def uy2(x):
    return 10*torch.ones(x.shape[0],1)


def uz1(x):
    return 10*x[:,2:3]*torch.exp(x[:,0:1]**2+x[:,1:2]**2+x[:,2:3]**2)


def uz2(x):
    return 10*torch.ones(x.shape[0],1)


# source term
def f1(x):
    return -a1*(30*torch.exp(x[:,0:1]**2+x[:,1:2]**2+x[:,2:3]**2)+20*(x[:,0:1]**2+x[:,1:2]**2+x[:,2:3]**2)*torch.exp(x[:,0:1]**2+x[:,1:2]**2+x[:,2:3]**2))


def f2(x):
    return torch.zeros(x.shape[0],1)


# parameters
def get_args():
    parser = argparse.ArgumentParser(description='Train the pde net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r1', '--range1', default=1.2, help='Range of initial weights', type=float)
    parser.add_argument('-r2', '--range2', default=0.8, help='Range of initial weights', type=float)
    parser.add_argument('-r3', '--range3', default=1, help='Range of initial weights', type=float)
    parser.add_argument('-r4', '--range4', default=0.8, help='Range of initial weights', type=float)
    parser.add_argument('-hx1', '--difference1', default=1e-6, help='difference parameter', type=float)
    parser.add_argument('-n1', '--numbers1', default=3375, help='numbers of the interior points', type=int)
    parser.add_argument('-n2', '--numbers2', default=225, help='number of the boundary points', type=int)
    parser.add_argument('-n3', '--numbers3', default=1350, help='number of the interface points', type=int)
    parser.add_argument('-n_int', '--numbers4', default=20, help='number of the integration points', type=int)
    parser.add_argument('-m', '--neurons', default=320, help='number of hidden layer nodes', type=int)
    parser.add_argument('-p1', '--penalty1', default=100, help='penalty parameter on boundary', type=float)
    parser.add_argument('-p2', '--penalty2', default=100, help='penalty parameter on interface', type=float)
    parser.add_argument('-alpha1', '--parameter1', default=1, help='coefficient1 of solution', type=float)
    parser.add_argument('-alpha2', '--parameter2', default=10, help='coefficient2 of solution', type=float)
    return parser.parse_args()


def main(r1, r2, r3, r4, hx1, N1, N2, N3, n_int, m, p1, p2, alpha1, alpha2, device):
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
    in_N = 3
    out_N = 1
    out_N2 = 3

    model1 = dnn1(in_N, m, out_N).to(device)                  # approximate real solution in Omega1
    model1 = model1.double()
    model1.apply(weights_init1)

    model2 = dnn1(in_N, m, out_N).to(device)                  # approximate real solution in Omega2
    model2 = model2.double()
    model2.apply(weights_init2)

    model3 = dnn2(in_N, m, out_N2).to(device)                 # approximate p1 = gradu1 in Omega1
    model3 = model3.double()
    model3.apply(weights_init3)

    model4 = dnn2(in_N, m, out_N2).to(device)                 # approximate p2 = gradu2 in Omega2
    model4 = model4.double()
    model4.apply(weights_init4)

    torch.set_default_tensor_type(torch.DoubleTensor)

    global a                                                   # domain(a,b) x (c,d) x (e,f)
    global b
    global c
    global d
    global e
    global f
    global a1
    global a2
    a = -1
    b = 1
    c = -1
    d = 1
    e = -1
    f = 1
    a1 = alpha1
    a2 = alpha2

    starttime = datetime.datetime.now()

    # interior euqations
    xy_inter = get_interior_point(N1)
    xy_inter = xy_inter.to(device)
    A1 = np.zeros((N1,8*m))
    A2 = np.zeros((N1,8*m))
    A3 = np.zeros((N1,8*m))
    A4 = np.zeros((N1,8*m))
    fv1 = np.zeros((N1,1))
    fv2 = np.zeros((N1,1))
    fv3 = np.zeros((N1,1))
    fv4 = np.zeros((N1,1))
    test = (xy_inter[:,0:1])**2 + (xy_inter[:,1:2])**2 + (xy_inter[:,2:3])**2
    for i in range(N1):
        if test[i] <= 0.75**2:
            gradxp = fdux(model3, xy_inter[i:i+1], hx1)
            gradyp = fduy(model3, xy_inter[i:i+1], hx1)
            gradzp = fduz(model3, xy_inter[i:i+1], hx1)
            gradxu = fdux(model1, xy_inter[i:i+1], hx1)
            gradyu = fduy(model1, xy_inter[i:i+1], hx1)
            gradzu = fduz(model1, xy_inter[i:i+1], hx1)
            A1[i,2*m:3*m] = -a1*gradxp.detach().cpu().numpy()
            A1[i,3*m:4*m] = -a1*gradyp.detach().cpu().numpy()
            A1[i,4*m:5*m] = -a1*gradzp.detach().cpu().numpy()
            A2[i,2*m:3*m] = model3(xy_inter[i:i+1])[0].detach().cpu().numpy()
            A2[i,0:m] = -gradxu.detach().cpu().numpy()
            A3[i,3*m:4*m] = model3(xy_inter[i:i+1])[0].detach().cpu().numpy()
            A3[i,0:m] = -gradyu.detach().cpu().numpy()
            A4[i,4*m:5*m] = model3(xy_inter[i:i+1])[0].detach().cpu().numpy()
            A4[i,0:m] = -gradzu.detach().cpu().numpy()
            fv1[i,0:1] = f1(xy_inter[i:i+1,:]).cpu().numpy()
        else:
            gradxp = fdux(model4, xy_inter[i:i+1], hx1)
            gradyp = fduy(model4, xy_inter[i:i+1], hx1)
            gradzp = fduz(model4, xy_inter[i:i+1], hx1)
            gradxu = fdux(model2, xy_inter[i:i+1], hx1)
            gradyu = fduy(model2, xy_inter[i:i+1], hx1)
            gradzu = fduz(model2, xy_inter[i:i+1], hx1)
            A1[i,5*m:6*m] = -a2*gradxp.detach().cpu().numpy()
            A1[i,6*m:7*m] = -a2*gradyp.detach().cpu().numpy()
            A1[i,7*m:8*m] = -a2*gradzp.detach().cpu().numpy()
            A2[i,5*m:6*m] = model4(xy_inter[i:i+1])[0].detach().cpu().numpy()
            A2[i,m:2*m] = -gradxu.detach().cpu().numpy()
            A3[i,6*m:7*m] = model4(xy_inter[i:i+1])[0].detach().cpu().numpy()
            A3[i,m:2*m] = -gradyu.detach().cpu().numpy()
            A4[i,7*m:8*m] = model4(xy_inter[i:i+1])[0].detach().cpu().numpy()
            A4[i,m:2*m] = -gradzu.detach().cpu().numpy()
            fv1[i,0:1] = f2(xy_inter[i:i+1,:]).cpu().numpy()

    # interface couple
    xy_interface = get_interface_point(N3)[0]
    xy_interface = xy_interface.to(device)
    n1 = get_interface_point(N3)[1]
    n2 = get_interface_point(N3)[2]
    n3 = get_interface_point(N3)[3]

    C = np.zeros((N3,8*m))
    fc = np.zeros((N3,1))
    uh_jump1 = model1(xy_interface)[0]
    uh_jump2 = -model2(xy_interface)[0]
    C[:,0:m] = uh_jump1.detach().cpu().numpy()
    C[:,m:2*m] = uh_jump2.detach().cpu().numpy()
    fc[:,0:1] = (u1(xy_interface) - u2(xy_interface)).cpu().numpy()

    D = np.zeros((N3,8*m))
    fd = np.zeros((N3,1))
    left_jump11 = a1*model3(xy_interface)[0]*n1
    left_jump12 = a1*model3(xy_interface)[0]*n2
    left_jump13 = a1*model3(xy_interface)[0]*n3
    left_jump21 = -a2*model4(xy_interface)[0]*n1
    left_jump22 = -a2*model4(xy_interface)[0]*n2
    left_jump23 = -a2*model4(xy_interface)[0]*n3
    right_jump1 = a1*(ux1(xy_interface)*n1 + uy1(xy_interface)*n2 + uz1(xy_interface)*n3)
    right_jump2 = - a2*(ux2(xy_interface)*n1 + uy2(xy_interface)*n2 + uz2(xy_interface)*n3)
    D[:, 2 * m:3 * m] = left_jump11.detach().cpu().numpy()
    D[:, 3 * m:4 * m] = left_jump12.detach().cpu().numpy()
    D[:, 4 * m:5 * m] = left_jump13.detach().cpu().numpy()
    D[:, 5 * m:6 * m] = left_jump21.detach().cpu().numpy()
    D[:, 6 * m:7 * m] = left_jump22.detach().cpu().numpy()
    D[:, 7 * m:8 * m] = left_jump23.detach().cpu().numpy()
    fd[:,0:1] = (right_jump1 + right_jump2).cpu().numpy()

    # boundary equations
    B = np.zeros((6*N2,8*m))
    xb = get_boundary_points0(N2)
    xb = xb.to(device)
    B[:, m:2*m] = model2(xb)[0].detach().cpu().numpy()
    fb = u2(xb).view([xb.size(0), 1]).cpu().numpy()

    # solving linear system
    matrix_a = np.vstack((A1, A2, A3, A4, p1*B, p2*C, p2*D))
    matrix_b = np.vstack((fv1, fv2, fv3, fv4, p1*fb, p2*fc, p2*fd))
    wu = np.linalg.lstsq(matrix_a, matrix_b)[0]
    wu1 = torch.transpose((torch.from_numpy(wu[0:m]).to(device)), 0, 1)
    wu2 = torch.transpose((torch.from_numpy(wu[m:2 * m]).to(device)), 0, 1)
    wu3 = torch.transpose((torch.from_numpy(wu[2 * m:3 * m]).to(device)), 0, 1)
    wu4 = torch.transpose((torch.from_numpy(wu[3 * m:4 * m]).to(device)), 0, 1)
    wu5 = torch.transpose((torch.from_numpy(wu[4 * m:5 * m]).to(device)), 0, 1)
    wu6 = torch.transpose((torch.from_numpy(wu[5 * m:6 * m]).to(device)), 0, 1)
    wu7 = torch.transpose((torch.from_numpy(wu[6 * m:7 * m]).to(device)), 0, 1)
    wu8 = torch.transpose((torch.from_numpy(wu[7 * m:8 * m]).to(device)), 0, 1)
    wuu1 = wu1
    wuu2 = wu2
    wup1 = torch.cat((wu3, wu4, wu5), dim=0)
    wup2 = torch.cat((wu6, wu7, wu8), dim=0)

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

    endtime = datetime.datetime.now()
    lasttime1 = (endtime - starttime).seconds
    lasttime2 = (endtime - starttime).microseconds
    print('Running time: %d.%ds' % (lasttime1, lasttime2))

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
    x[:, 2] = (f - e) / 2 * xr[:, 2] + (e + f) / 2
    test2 = x[:,0:1]**2 + x[:,1:2]**2 + x[:,2:3]**2
    output_u = torch.zeros(n_int ** 3,1)
    u_exact = torch.zeros(n_int ** 3,1)
    output_p1 = torch.zeros(n_int ** 3, 1)
    output_p2 = torch.zeros(n_int ** 3, 1)
    output_p3 = torch.zeros(n_int ** 3, 1)
    p1_exact = torch.zeros(n_int ** 3, 1)
    p2_exact = torch.zeros(n_int ** 3, 1)
    p3_exact = torch.zeros(n_int ** 3, 1)
    for i in range(n_int ** 3):
        if test2[i] <= 0.75 ** 2:
            output_u[i:i + 1, 0:1] = model1(x[i:i + 1, :])[1]
            u_exact[i:i + 1, 0:1] = u1(x[i:i + 1, :])
            output_p1[i:i + 1, 0:1] = model3(x[i:i + 1, :])[1]
            output_p2[i:i + 1, 0:1] = model3(x[i:i + 1, :])[2]
            output_p3[i:i + 1, 0:1] = model3(x[i:i + 1, :])[3]
            p1_exact[i:i + 1, 0:1] = ux1(x[i:i + 1, :])
            p2_exact[i:i + 1, 0:1] = uy1(x[i:i + 1, :])
            p3_exact[i:i + 1, 0:1] = uz1(x[i:i + 1, :])
        else:
            output_u[i:i + 1, 0:1] = model2(x[i:i + 1, :])[1]
            u_exact[i:i + 1, 0:1] = u2(x[i:i + 1, :])
            output_p1[i:i + 1, 0:1] = model4(x[i:i + 1, :])[1]
            output_p2[i:i + 1, 0:1] = model4(x[i:i + 1, :])[2]
            output_p3[i:i + 1, 0:1] = model4(x[i:i + 1, :])[3]
            p1_exact[i:i + 1, 0:1] = ux2(x[i:i + 1, :])
            p2_exact[i:i + 1, 0:1] = uy2(x[i:i + 1, :])
            p3_exact[i:i + 1, 0:1] = uz2(x[i:i + 1, :])
    err_L2 = torch.sqrt((b - a) * (d - c) * (f - e) / 8 * torch.sum(torch.pow(output_u - u_exact, 2) * wr))
    err_L2_p = torch.sqrt((b - a) * (d - c) * (f - e) / 8 * torch.sum((torch.pow(output_p1 - p1_exact, 2)+torch.pow(output_p2 - p2_exact, 2)+torch.pow(output_p3 - p3_exact, 2)) * wr))
    u_L2 = torch.sqrt((b - a) * (d - c) * (f - e) / 8 * torch.sum(torch.pow(u_exact, 2) * wr))
    p_L2 = torch.sqrt((b - a) * (d - c) * (f - e) / 8 * torch.sum((torch.pow(p1_exact, 2)+torch.pow(p2_exact, 2)+torch.pow(p3_exact, 2)) * wr))
    err_rL2 = err_L2 / u_L2
    err_rL2_p = err_L2_p / p_L2
    print('Final L^2 error:', err_L2.item())
    print('Final u_L^2 norm:', u_L2.item())
    print('Final L^2 relative error:', err_rL2.item())
    print('Final p_L^2 error:', err_L2_p.item())
    print('Final p_u_L^2 norm:', p_L2.item())
    print('Final p_L^2 relative error:', err_rL2_p.item())
    results = torch.zeros(1, 2)
    results[0, 0] = err_L2
    results[0, 1] = err_L2_p
    return results

    # # pictures of numerical solution, real solution and error
    # with torch.no_grad():
    #     x1 = torch.linspace(a, b, 300)
    #     x2 = torch.linspace(c, d, 300)
    #     Y, X = torch.meshgrid(x2, x1)  # indexing='xy' or 'ij'
    #     Z = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    #     Z = Z.to(device)
    #     zz = torch.zeros(90000,1)
    #     zzz = torch.cat((Z,zz),dim=1)
    #     pred1 = torch.zeros(90000, 1)
    #     pred2 = torch.zeros(90000, 1)
    #     pred3 = torch.zeros(90000, 1)
    #     test3 = Z[:,0:1] ** 2 + Z[:,1:2] ** 2
    #     for i in range(90000):
    #         if test3[i] <= 0.75**2:
    #             pred1[i, 0] = model1(zzz[i:i+1,:])[1]
    #             pred2[i, 0] = u1(zzz[i:i+1,:])
    #             pred3[i, 0] = torch.abs(model1(zzz[i:i + 1, :])[1] - u1(zzz[i:i + 1,:]))
    #         else:
    #             pred1[i, 0] = model2(zzz[i:i+1,:])[1]
    #             pred2[i, 0] = u2(zzz[i:i + 1, :])
    #             pred3[i, 0] = torch.abs(model2(zzz[i:i + 1, :])[1] - u2(zzz[i:i + 1, :]))
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
#     args = get_args()
#     device = torch.device('cpu')
#     main(r1=args.range1, r2=args.range2, r3=args.range3, r4=args.range4, hx1=args.difference1, N1=args.numbers1,
#          N2=args.numbers2, N3=args.numbers3, n_int=args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha1=args.parameter1,
#          alpha2=args.parameter2, device=device)

# # change parameters
# if __name__ == '__main__':
#     data = np.zeros((15,1))
#     args = get_args()  #Change parameters
#     device = torch.device('cpu')
#     for i in range(15):
#         # for j in range(20):
#         r1 = 1.2
#         r2 = 0.8
#         r3 = 1
#         r4 = 0.8
#         data[i,0] = main(r1, r2, r3, r4, hx1=args.difference1, N1=args.numbers1,
#              N2=args.numbers2, N3=args.numbers3, n_int=args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha1=args.parameter1,
#              alpha2=args.parameter2, device=device)
#     print(data)


# run ten times and give the average value
if __name__ == '__main__':
    args = get_args()
    data = torch.zeros(11,2)
    device = torch.device('cpu')
    aaa = 0
    bbb = 0
    for i in range(10):
        data[i,:] = main(r1=args.range1, r2=args.range2,r3=args.range3,r4=args.range4, hx1=args.difference1, N1=args.numbers1,
             N2=args.numbers2, N3=args.numbers3, n_int=args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha1=args.parameter1,
             alpha2=args.parameter2, device=device)
        aaa += data[i,0]
        bbb += data[i,1]
    data[10,0] = aaa/10
    data[10,1] = bbb/10
    print('%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e ' % (data[0,0],data[1,0],data[2,0],data[3,0],data[4,0],data[5,0],data[6,0],data[7,0],data[8,0],data[9,0],data[10,0]))
    print('\n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e ' % (data[0,1],data[1,1],data[2,1],data[3,1],data[4,1],data[5,1],data[6,1],data[7,1],data[8,1],data[9,1],data[10,1]))
    # print(data)



