from __future__ import print_function

import argparse
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import math
import numpy as np
import torch.nn.functional as F
from utils import Logger, mkdir_p
from scipy.io import loadmat
from numpy.polynomial.legendre import leggauss
import hdf5storage
from torch.nn.parameter import Parameter

parser = argparse.ArgumentParser(description='PyTorch 3D manifold PDE training')
# Datasets
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--width', default=50, type=int, help='width of training data')
# Optimization options
parser.add_argument('--epoch', default=1, type=int, metavar='N',
                    help='number of epochs to run')
parser.add_argument('--iters', default=30000, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--function', default='resnet', type=str,
                    help='function to approximate')
parser.add_argument('--activation', default='relu', type=str,
                    help='activation')
parser.add_argument('--optim', default='adam', type=str,
                    help='function to approximate')
# network setting
parser.add_argument('--m', default=100, type=int, help='the width of neuron size')
parser.add_argument('--layers', default=4, type=int, help='')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--trial', type=int, help='trial number')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
# Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
# function type
parser.add_argument('--penalty', default=0.0, type=float, help='penalty for nn')
# region
parser.add_argument('--left', default=-3, type=float, help='left boundary of square region')
parser.add_argument('--right', default=3, type=float, help='right boundary of square region')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = 1  # set the random seed to be 1 for reproducibility
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)

if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

class sin_poly2_params(nn.Module):
    def __init__(self, num_features, a=0, a1=0.01, b=0, b1=0.01, c=1, c1=0.01, d=1, d1=0.01):
        super(sin_poly2_params, self).__init__()
        # self.relu = nn.ReLU(inplace=True)
        self.a = Parameter(torch.Tensor(1, num_features))
        self.b = Parameter(torch.Tensor(1, num_features))
        self.c = Parameter(torch.Tensor(1, num_features))
        self.d = Parameter(torch.Tensor(1, num_features))

        # initialization ## good for image fitting
        torch.nn.init.normal_(self.a, mean=a, std=a1)
        torch.nn.init.normal_(self.b, mean=b, std=b1)
        torch.nn.init.normal_(self.c, mean=c, std=c1)
        torch.nn.init.normal_(self.d, mean=d, std=d1)
        # torch.nn.init.constant_(self.a, a)
        # torch.nn.init.constant_(self.b, b)
        # torch.nn.init.constant_(self.c, c)
        # torch.nn.init.constant_(self.d, d)
        # print('a 0 and b 1 initialization')
        print('sin poly2  ax-{} bx2-{} c-{} sin( {} x)'.format(a, b, c, d))
    def forward(self, x):
        y = x * self.a.expand_as(x)
        y = y + self.b.expand_as(x) * x ** 2
        y = y + self.c.expand_as(x) * torch.sin(self.d.expand_as(x)*x)
        # y = self.c.expand_as(x) * torch.sin(self.d.expand_as(x)*x)
        return y

class one_layer(nn.Module):
    def __init__(self, m):
        super(one_layer, self).__init__()
        self.fc1 = nn.Linear(m, m)
        if args.activation == 'relu':
            self.act = nn.ReLU(inplace=True)
            print('use relu activation')
        elif args.activation == 'sigmoid':
            self.act = torch.sigmoid
            print('use sigmoid activation')
        elif args.activation == 'tanh':
            self.act = torch.tanh
            print('use tanh activation')
        elif args.activation == 'softplus':
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
        self.layers = self._make_layer(m, args.layers - 2)
        self.output = nn.Linear(m, 1, bias=True)
        if args.activation == 'relu':
            self.relu = nn.ReLU(inplace=True)
            print('use relu activation')
        elif args.activation == 'sigmoid':
            self.relu = torch.sigmoid
            print('use sigmoid activation')
        elif args.activation == 'tanh':
            self.relu = torch.tanh
            print('use tanh activation')
        elif args.activation == 'softplus':
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
        x = (x - args.left) * 2 / (args.right - args.left) + (-1)  # normalize to [-1,1]

        x = self.fc1(x)
        x = self.relu(x)
        x = self.layers(x)
        x = self.output(x)

        return x


## points from 2D to 3D manifold
def points(theta):
    theta1 = theta[:, 0:1]
    theta2 = theta[:, 1:2]
    x1 = (2 + torch.cos(theta1)) * torch.cos(theta2)
    x2 = (2 + torch.cos(theta1)) * torch.sin(theta2)
    x3 = torch.sin(theta1)
    return torch.cat((x1, x2, x3), 1)


'''
HyperParams Setting for Network
'''
m = args.m  # number of hidden size

## gaussian points
degree = 50
x, wx = leggauss(degree)
y, wy = leggauss(degree)

gaussValues2DQuad_pt = np.zeros((degree ** 2, 2))
gaussValues2DQuad_w = np.zeros((degree ** 2, 1))

for i in range(degree):
    for j in range(degree):
        gaussValues2DQuad_pt[i * degree + j, 0] = x[i]
        gaussValues2DQuad_pt[i * degree + j, 1] = y[j]
        gaussValues2DQuad_w[i * degree + j] = wx[i] * wy[j]

left = 0
right = 2 * math.pi

new_gaussValues2DQuad_pt = (right - left) / 2 * (gaussValues2DQuad_pt + 1) + left
jacobi = ((right - left) / 2) ** 2

new_gaussValues2DQuad_pt_torch = torch.cuda.FloatTensor(new_gaussValues2DQuad_pt)
gaussValues2DQuad_w_torch = torch.cuda.FloatTensor(gaussValues2DQuad_w)


def main():
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    with open(args.checkpoint + "/Config.txt", 'w+') as f:
        for (k, v) in args._get_kwargs():
            f.write(k + ' : ' + str(v) + '\n')

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Model
    if args.function == 'fc_activate':
        model = FcNet_activate(m)
    else:
        model = None

    model = model.cuda()
    cudnn.benchmark = True

    print('Total params: %.6f' % (sum(p.numel() for p in model.parameters())))
    """
    Define Residual Methods and Optimizer
    """
    criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)

    # Resume
    title = ''
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Losses', 'MSE with GT', 'integral'])
    # Train and val
    if args.width <= 140:
        data_dict = loadmat('training_data/manifold3D_N' + str(args.width) + '.mat')
    else:
        data_dict = hdf5storage.loadmat('training_data/manifold3D_N' + str(args.width) + '.mat')
    x = torch.cuda.FloatTensor(data_dict['y'])
    u = torch.cuda.FloatTensor(data_dict['u'])
    fc = torch.cuda.FloatTensor(data_dict['f'] / data_dict['c'])
    L = torch.cuda.FloatTensor(data_dict['L'])
    batch = (L, fc)
    print('x size: ', x.size(), ' L size:', L.size())
    print('x max: ', torch.max(x), '-- x min: ', torch.min(x))
    print('forward error: ', criterion(torch.mm(L, u), fc))

    total_iters = 0
    for _ in range(args.epoch):
        for iter in range(args.iters):
            lr = cosine_lr(optimizer, args.lr, total_iters, args.iters * args.epoch)
            losses, mse, int = train(batch, x, u, model, criterion, optimizer, use_cuda, total_iters, lr)
            logger.append([lr, losses, mse, int])
            total_iters += 1
    # save model
    save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                    checkpoint=args.checkpoint)

    logger.close()
    logger.plot()


def train(batch, x, u_true, model, criterion, optimizer, use_cuda, iter, lr):
    suffix = ''
    with torch.no_grad():
        model.eval()
        u_nn = model(points(new_gaussValues2DQuad_pt_torch))
        integral = torch.sqrt(torch.sum(jacobi * gaussValues2DQuad_w_torch * (u_nn ** 2)))

        u_nn = model(x)
        MSE = criterion(u_nn, u_true).item()
        integral = integral.item()
        suffix = suffix + 'MSE GT: {diff:.4f}, Abs {abs:.4f}| Integral: {int:.4f}|'.format(diff=MSE, int=integral,
                                                                                           abs=torch.max(torch.abs(
                                                                                               u_nn - u_true)))

    # switch to train mode
    model.train()
    end = time.time()
    '''
    points sampling
    '''
    L, fc = batch
    u = model(x)
    embedded_dev = torch.mm(L, u)
    loss = criterion(embedded_dev, fc)

    if args.penalty > 0:
        loss = loss + args.penalty * torch.mean(u ** 2)

    # compute gradient and do optim step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time = time.time() - end

    suffix = 'iter: {iter:.0f}|lr: {lr:.4f}| Batch: {bt:.2f}s | training MSE: {mse: .5f} |'.format(
        bt=batch_time, iter=iter, lr=lr, mse=loss.item()) + suffix
    print(suffix)

    return loss.item(), MSE, integral

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def scheduler(opt, base_lr, e, epochs):
    schedule = [int(epochs / 4 * 3)]
    if e in schedule:
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    for param_group in opt.param_groups:
        current_lr = param_group['lr']
    return current_lr

def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * (base_lr) * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    main()
    print('******************* evaluation ********************')
    os.system('python relative_error.py --N ' + str(args.width) + ' --checkpoint ' + args.checkpoint + ' --m ' + str(args.m) + ' --trial ' + str(args.trial) + ' --activation '+ args.activation)
