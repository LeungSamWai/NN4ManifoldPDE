import numpy as np
from numpy.polynomial.legendre import leggauss
import math
import matplotlib.pyplot as plt

from torch import nn
import torch
from scipy.io import loadmat
import argparse
import hdf5storage
from torch.nn.parameter import Parameter

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--N', type=int)
parser.add_argument('--m', type=int)
parser.add_argument('--trial', type=int)
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--activation', default='relu', type=str, help='activation')
args = parser.parse_args()

def p_true(theta):
    theta1 = theta[:,0:1]
    theta2 = theta[:,1:2]
    beta1 = 1
    alpha1 = 1
    k = 2
    a = 2
    true = (alpha1 * torch.sin(k * theta2) - beta1/(a+torch.cos(theta1))*k* torch.cos(k * theta2))*torch.cos(theta1)
    return true

def points(theta):
    theta1 = theta[:, 0:1]
    theta2 = theta[:, 1:2]
    x1 = (2+torch.cos(theta1))*torch.cos(theta2)
    x2 = (2+torch.cos(theta1))*torch.sin(theta2)
    x3 = torch.sin(theta1)
    return torch.cat((x1, x2, x3), 1)

class sin_poly2_params(nn.Module):
    def __init__(self, num_features, a=0, a1=0.01, b=0, b1=0.01, c=1, c1=0.01, d=1, d1=0.01):
        super(sin_poly2_params, self).__init__()
        # self.relu = nn.ReLU(inplace=True)
        self.a = Parameter(torch.Tensor(1, num_features))
        self.b = Parameter(torch.Tensor(1, num_features))
        self.c = Parameter(torch.Tensor(1, num_features))
        self.d = Parameter(torch.Tensor(1, num_features))

        # ## initialization ## good for image fitting
        torch.nn.init.normal_(self.a, mean=a, std=a1)
        torch.nn.init.normal_(self.b, mean=b, std=b1)
        torch.nn.init.normal_(self.c, mean=c, std=c1)
        torch.nn.init.normal_(self.d, mean=d, std=d1)
        # print('a 0 and b 1 initialization')
        print('sin poly2  ax-{} bx2-{} c-{} sin( {} x)'.format(a, b, c, d))
    def forward(self, x):
        y = x * self.a.expand_as(x)
        y = y + self.b.expand_as(x) * x ** 2
        y = y + self.c.expand_as(x) * torch.sin(self.d.expand_as(x)*x)
        return y

class sin_gaussian_params(nn.Module):
    def __init__(self, num_features, a=0.5, a1=0.01, b=2, b1=0.01, c=0.01, c1=0.001, d=0.1, d1=0.001):
        super(sin_gaussian_params, self).__init__()
        # self.relu = nn.ReLU(inplace=True)
        self.a = Parameter(torch.Tensor(1, num_features))
        self.b = Parameter(torch.Tensor(1, num_features))
        self.c = Parameter(torch.Tensor(1, num_features))
        self.d = Parameter(torch.Tensor(1, num_features))
        # ## initialization ## good for image fitting
        torch.nn.init.normal_(self.a, mean=a, std=a1)
        torch.nn.init.normal_(self.b, mean=b, std=b1)
        torch.nn.init.normal_(self.c, mean=c, std=c1)
        torch.nn.init.normal_(self.d, mean=d, std=d1)
        print('sin gaussian {} {} {} {} {} {} {} {}'.format(a, a1, b, b1, c, c1, d, d1))

    def forward(self, x):
        y = self.a.expand_as(x) * torch.sin(self.b.expand_as(x) * x)
        y = y + self.c.expand_as(x)*torch.exp(-x**2/(2*(self.d.expand_as(x)**2)))
        return y

class one_layer(nn.Module):
    def __init__(self, m):
        super(one_layer, self).__init__()
        self.fc1 = nn.Linear(m, m)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
            print('use relu activation')
        elif activation == 'sigmoid':
            self.act = torch.sigmoid
            print('use sigmoid activation')
        elif activation == 'tanh':
            self.act = torch.tanh
            print('use tanh activation')
        elif activation == 'softplus':
            self.act = torch.nn.Softplus()
            print('use softplus activation')
        elif args.activation == 'raf':
            self.act = sin_poly2_params(m)
            print('use raf activation')
        elif args.activation == 'raf_sg':
            self.act = sin_gaussian_params(m)
            print('use sg raf activation')
    def forward(self, x):
        y = self.fc1(x)
        y = self.act(y)
        return y

class FcNet_activate(nn.Module):
    '''
    fully connected neural network
    '''
    def __init__(self, m):
        super(FcNet_activate, self).__init__()
        self.fc1 = nn.Linear(3, m)
        self.layers = self._make_layer(m, layers-2)
        self.output = nn.Linear(m, 1, bias=True)
        if activation == 'relu':
            self.relu = nn.ReLU(inplace=True)
            print('use relu activation')
        elif activation == 'sigmoid':
            self.relu = torch.sigmoid
            print('use sigmoid activation')
        elif activation == 'tanh':
            self.relu = torch.tanh
            print('use tanh activation')
        elif activation == 'softplus':
            self.relu = torch.nn.Softplus()
            print('use softplus activation')
        elif args.activation == 'raf':
            self.relu = sin_poly2_params(m)
            print('use raf activation')
        elif args.activation == 'raf_sg':
            self.relu = sin_gaussian_params(m)
            print('use sg raf activation')

    def _make_layer(self, m, n_layers):
        layers = []
        for i in range(n_layers):
            layers.append(one_layer(m))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = (x - left_nn) * 2 / (right_nn - left_nn) + (-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.layers(x)
        x = self.output(x)
        return x

'''
Params Setting for Function
'''

left_nn = -6
right_nn = 6

left = 0
right = 2 * math.pi

activation = args.activation
layers = 4
m = args.m
N = args.N
trial = args.trial
file_name = args.checkpoint + '/checkpoint.pth.tar'
print('loading the ckpt: {}'.format(file_name))
model = FcNet_activate(m).cuda()
model.load_state_dict(torch.load(file_name)['state_dict'])


degree = 300
x, wx = leggauss(degree)
y, wy = leggauss(degree)

gaussValues2DQuad_pt = np.zeros((degree**2, 2))
gaussValues2DQuad_w = np.zeros((degree**2, 1))

for i in range(degree):
    for j in range(degree):
        gaussValues2DQuad_pt[i*degree+j,0] = x[i]
        gaussValues2DQuad_pt[i*degree+j,1] = y[j]
        gaussValues2DQuad_w[i*degree+j] = wx[i]*wy[j]

new_gaussValues2DQuad_pt= (right - left)/2*(gaussValues2DQuad_pt+1)+left
jacobi = ((right-left)/2)**2
print(new_gaussValues2DQuad_pt[:30])

new_gaussValues2DQuad_pt_torch = torch.cuda.FloatTensor(new_gaussValues2DQuad_pt)
gaussValues2DQuad_w_torch = torch.cuda.FloatTensor(gaussValues2DQuad_w)

u_true = p_true(new_gaussValues2DQuad_pt_torch).detach().cpu().numpy()
u_nn = model(points(new_gaussValues2DQuad_pt_torch)).detach().cpu().numpy()
print(points(new_gaussValues2DQuad_pt_torch).size())
L2_p_true = np.sqrt(np.sum(jacobi*gaussValues2DQuad_w*np.square(u_true)))
print('L2_p_true', L2_p_true)
L2_p_nn = np.sqrt(np.sum(jacobi*gaussValues2DQuad_w*np.square(u_nn)))
print('L2_p_nn', L2_p_nn)
L2_error = np.sqrt(np.sum(jacobi*gaussValues2DQuad_w*np.square(u_nn - u_true)))
L2_error_relative = L2_error/L2_p_true
print('relative l2 error:', L2_error_relative)

fig = plt.figure(figsize=(6, 5))
ax1 = fig.add_subplot(1, 1, 1)
x = new_gaussValues2DQuad_pt[:,0].reshape((degree, degree))
y = new_gaussValues2DQuad_pt[:,1].reshape((degree, degree))
u_nn = u_nn.reshape((degree, degree))
u_true = u_true.reshape((degree, degree))
c = ax1.pcolor(x, y, np.abs(u_nn - u_true))
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
print('L infinity on gaussian pts, ', np.max(np.abs(u_nn - u_true)))
guassion_pts_error = np.max(np.abs(u_nn - u_true))

if N <= 140:
    data_dict = loadmat('training_data/manifold3D_N' + str(N) + '.mat')
else:
    data_dict = hdf5storage.loadmat('training_data/manifold3D_N' + str(N) + '.mat')

train_pts = torch.cuda.FloatTensor(data_dict['y'])
value_on_train_pts = data_dict['u']
nn_model_on_train_pts = model(train_pts).detach().cpu().numpy()
print('L infinity on train pts, ', np.max(np.abs(nn_model_on_train_pts - value_on_train_pts)))
train_pts_error = np.max(np.abs(nn_model_on_train_pts - value_on_train_pts))

if N <= 140:
    inverse_value_on_train_pts = data_dict['uhat']
    print('inverse error:', np.max(np.abs(value_on_train_pts-inverse_value_on_train_pts)))
    print('diff pseudo-inverse and nn on training point:', np.max(np.abs(nn_model_on_train_pts-inverse_value_on_train_pts)))
    diff = np.max(np.abs(nn_model_on_train_pts-inverse_value_on_train_pts))
    f = open('N'+str(N)+'.txt', "a")
    f.write('{}   {}   {}   {}   {}\n'.format(trial, L2_error_relative, guassion_pts_error, train_pts_error, diff))
    f.close()
else:
    f = open('N' + str(N) + '.txt', "a")
    f.write('{}   {}   {}   {}  \n'.format(trial, L2_error_relative, guassion_pts_error, train_pts_error))
    f.close()