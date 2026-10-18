"""LRaNN-FDM paper experiment; see docs/reproduction.md for settings."""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
import lrann_interface.compat

# Solve High-dimensional Elliptic Interface Problem (example4.4)

import torch
import torch.nn as nn
import numpy as np
import argparse
import time

# torch.manual_seed(20)

# define FNN
class dnn(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=True)                 # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)                # output layer
        self.phi = torch.tanh                                                         # activation function

    def forward(self, x):
        x = self.phi(self.hidden(x))                                                  # output of hidden layer
        y = self.predict(x)                                                           # linear output
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
# model: function, xr: sampling points, h: difference step, d: dimension, m: neurons
# Laplace operator
def laplace_u(model, xr, h, d, m):
    rr = torch.zeros(xr.shape[0], m)
    for i in range(d):
        ei = torch.zeros_like(xr)
        ei[:,i:i+1] = 1
        rr += (model(xr+h*ei)[0] + model(xr-h*ei)[0] - 2*model(xr)[0])/(h**2)
    return rr


# gradient operator
def gradient_u(model, xr, h, d, m):
    gradu = torch.zeros(xr.shape[0], d, m)
    for i in range(d):
        ei = torch.zeros_like(xr)
        ei[:, i:i+1] = 1
        gradu[:,i,:] = (model(xr+h*ei)[0] - model(xr-h*ei)[0])/(2*h)
    return gradu


# first-derivative with respect to x
from lrann_interface.finite_difference import fdux_variant3 as fdux


# integration points(MC_method)
def get_monte_points(N,d):
    xb = (b - a) * torch.rand(N, d) + a
    return xb


# sampling points
# boundary points
def get_boundary_points0(N, d):
    xb = (b-a)*torch.rand(N*2*d, d) + a
    for i in range(d):
        xb[N * (2 * i):N * (2 * i + 1), i:i + 1] = a*torch.ones(N,1)
        xb[N * (2 * i + 1):N * (2 * i + 2), i:i + 1] = b*torch.ones(N,1)
    return xb


# interior points
def get_interior1_point(N, d):
    xb = (b-a)/2*torch.rand(N, d) + a
    return xb


def get_interior2_point(N, d):
    xb = (b-a)/2*torch.rand(N, d) + a + (b-a)/2
    return xb


# interface points
def get_interface_point(N, d):
    xb = (b-a)*torch.rand(N,d) + a
    xb[:,0:1] = 0.5
    return xb


# real solution
# solution in Omega1
def u1(x, d):
    rr = torch.zeros(x.shape[0],1)
    for i in range(d):
        rr += x[:,i:i+1]**2
    r = 1/d*rr
    return r


# solution in Omega2
def u2(x, d):
    rr = torch.zeros(x.shape[0],1)
    for i in range(d):
        rr += x[:,i:i+1]
    r = 1/d*rr
    return r


# partial derivative of real solution
def gradu1(x,d):
    return 2/d*x


def gradu2(x,d):
    return torch.ones(x.shape[0],d)/d


def u1x(x,d):
    return 1/d*2*x[:,0:1]


def u2x(x,d):
    return 1/d*torch.ones(x.shape[0],1)


# source term
def f1(x):
    return -2*a1*torch.ones(x.shape[0],1)


def f2(x):
    return torch.zeros(x.shape[0],1)


# parameters
def get_args():
    parser = argparse.ArgumentParser(description='Train the pde net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dim', '--dimension', default=20, help='dimension of the interface problem', type=int)
    parser.add_argument('-r1', '--range1',  default=0.01, help='Range of initial weights', type=float)
    parser.add_argument('-r2', '--range2', default=0.01, help='Range of initial weights', type=float)
    parser.add_argument('-hx1', '--difference1', default=1e-6, help='difference parameter', type=float)
    parser.add_argument('-hx2', '--difference2', default=5e-4, help='difference parameter', type=float)
    parser.add_argument('-n1', '--numbers1', default=10, help='numbers of the interior points', type=int)             # N/2 （N=20）
    parser.add_argument('-n_int', '--numbers4', default=10000, help='number of the integration points', type=int)
    parser.add_argument('-m', '--neurons', default=1800, help='number of hidden layer nodes', type=int)
    parser.add_argument('-p1', '--penalty1', default=100, help='penalty parameter on boundary', type=float)
    parser.add_argument('-p2', '--penalty2', default=100, help='penalty parameter on boundary', type=float)
    parser.add_argument('-alpha1', '--parameter1', default=1, help='coefficient1 of solution', type=float)
    parser.add_argument('-alpha2', '--parameter2', default=1, help='coefficient2 of solution', type=float)
    parser.add_argument('-m_d', '--parameter3', default=6, help='parameter of sampling', type=int)
    return parser.parse_args()


def main(dim, r1, r2, hx1, hx2, N1, n_int, m, p1, p2, alpha1, alpha2, m_d, device):

    # Set network parameters
    global ra1
    global rb1
    global ra2
    global rb2
    global d
    d = dim
    ra1 = -r1
    rb1 = r1
    ra2 = -r2
    rb2 = r2
    in_N = d
    out_N = 1

    model1 = dnn(in_N, m, out_N).to(device)                                      # approximate real solution in Omega1
    model1 = model1.double()
    model1.apply(weights_init1)

    model2 = dnn(in_N, m, out_N).to(device)                                      # approximate real solution in Omega2
    model2 = model2.double()
    model2.apply(weights_init2)

    torch.set_default_tensor_type(torch.DoubleTensor)

# Set parameters

    global a                                                                     # domain(a,b)^d
    global b
    global a1
    global a2
    a = 0
    b = 1
    a1 = alpha1
    a2 = alpha2

    start = time.perf_counter()

    # interior euqations
    xy_inter1 = get_interior1_point(N1*m_d*d,d)
    xy_inter1 = xy_inter1.to(device)
    xy_inter2 = get_interior2_point(N1*m_d*d,d)
    xy_inter2 = xy_inter2.to(device)

    A = np.zeros((2*N1*m_d*d,2*m))
    fv = np.zeros((2*N1*m_d*d,1))

    A[0:N1*m_d*d,0:m] = -a1*laplace_u(model1,xy_inter1,hx2,d,m).detach().cpu().numpy()
    fv[0:N1*m_d*d,:] = f1(xy_inter1).cpu().numpy()

    A[N1*m_d*d:2*N1*m_d*d,m:2*m] = -a2*laplace_u(model2,xy_inter2,hx2,d,m).detach().cpu().numpy()
    fv[N1*m_d*d:2*N1*m_d*d,:] = f2(xy_inter2).cpu().numpy()

    # interface couple
    xy_interface = get_interface_point(m_d*2*d*d,d)
    xy_interface = xy_interface.to(device)

    C = np.zeros((m_d*2*d*d,2*m))
    fc = np.zeros((m_d*2*d*d,1))
    uh_jump1 = model1(xy_interface)[0]
    uh_jump2 = -model2(xy_interface)[0]
    C[:,0:m] = (uh_jump1).detach().cpu().numpy()
    C[:,m:2*m] = (uh_jump2).detach().cpu().numpy()
    fc[:,0:1] = (u1(xy_interface,d) - u2(xy_interface,d)).cpu().numpy()

    D = np.zeros((m_d*2*d*d,2*m))
    fd = np.zeros((m_d*2*d*d,1))
    D[:,0:m] = a1*fdux(model1,xy_interface,hx1).detach().cpu().numpy()
    D[:,m:2*m] = -a2*fdux(model2,xy_interface,hx1).detach().cpu().numpy()
    fd[:,0:1] = (a1*u1x(xy_interface,d) - a2*u2x(xy_interface,d)).cpu().numpy()

    # boundary equations
    B = np.zeros((m_d*2*d*d,2*m))
    fb = np.zeros((m_d*2*d*d,1))
    xb = get_boundary_points0(m_d*d,d)
    xb = xb.to(device)
    test1 = xb[:,0:1]
    for i in range(m_d*2*d*d):
        if test1[i] >= 0.5:
            B[i,0:m] = (model1(xb[i:i+1,:])[0]).detach().cpu().numpy()
            fb[i,0:1] = (u1(xb[i:i+1,:],d)).cpu().numpy()
        else:
            B[i,m:2*m] = (model2(xb[i:i+1,:])[0]).detach().cpu().numpy()
            fb[i,0:1] = (u2(xb[i:i+1,:],d)).cpu().numpy()


    # solving linear system
    matrix_a = np.vstack((A, p1*B, p2*C, p2*D))
    matrix_b = np.vstack((fv, p1*fb, p2*fc, p2*fd))
    wu = np.linalg.lstsq(matrix_a, matrix_b)[0]
    wu = torch.from_numpy(wu)
    wu = wu.to(device)
    wu = torch.transpose(wu, 0, 1)
    wu1 = wu[0,0:m]
    wu2 = wu[0,m:2*m]

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
    x = get_monte_points(n_int,d)
    x = x.to(device)
    x.requires_grad_()
    test2 = x[:,0:1]
    output_u = torch.zeros(n_int, 1)
    u_exact = torch.zeros(n_int, 1)
    for i in range(n_int):
        if test2[i] >= 0.5:
            output_u[i:i + 1, 0:1] = model1(x[i:i + 1, :])[1]
            u_exact[i:i + 1, 0:1] = u1(x[i:i + 1, :],d)
        else:
            output_u[i:i + 1, 0:1] = model2(x[i:i + 1, :])[1]
            u_exact[i:i + 1, 0:1] = u2(x[i:i + 1, :],d)
    err_L2 = torch.sqrt((b-a)**d * 1/n_int * torch.sum(torch.pow(output_u - u_exact, 2)))
    u_L2 = torch.sqrt((b-a)**d * 1/n_int * torch.sum(torch.pow(u_exact, 2)))
    err_rL2 = err_L2 / u_L2
    print('Final L^2 error:', err_L2.item())
    print('Final u_L^2 norm:', u_L2.item())
    print('Final L^2 relative error:', err_rL2.item())


    return err_rL2


# if __name__ == '__main__':
#     args = get_args()
#     device = torch.device('cpu')
#     main(dim=args.dimension, r1=args.range1, r2=args.range2, hx1=args.difference1, hx2=args.difference2, N1=args.numbers1,
#          n_int=args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha1=args.parameter1,
#          alpha2=args.parameter2, m_d=args.parameter3, device=device)


# run ten times and give the average value
if __name__ == '__main__':
    args = get_args()  # Change parameters
    data = np.zeros((11,1))
    device = torch.device('cpu')
    aaa = 0
    for i in range(10):
        data[i,0] = main(dim=args.dimension, r1=args.range1, r2=args.range2, hx1=args.difference1, hx2=args.difference2, N1=args.numbers1,
         n_int=args.numbers4, m=args.neurons, p1=args.penalty1, p2=args.penalty2, alpha1=args.parameter1,
         alpha2=args.parameter2, m_d=args.parameter3, device=device)
        aaa += data[i, 0]
    data[10, 0] = aaa / 10
    print('%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e \n%e ' % (data[0, 0], data[1, 0], data[2, 0], data[3, 0], data[4, 0], data[5, 0], data[6, 0], data[7, 0], data[8, 0],data[9, 0], data[10, 0]))



