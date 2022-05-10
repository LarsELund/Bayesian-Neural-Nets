#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parameter_init(low,high,size):
    random_init = (low - high) * torch.rand(size,device = DEVICE) + high
    return random_init  

class PropagateFlow(nn.Module):
    def __init__(self,transform,dim,num_transforms):
        super().__init__() 
        if transform == 'Planar':
            self.transforms = nn.ModuleList([PlanarTransform(dim) for i in range(num_transforms)])
        elif transform == 'Radial':
            self.transforms = nn.ModuleList([RadialTransform(dim) for i in range(num_transforms)])
        elif transform == 'Sylvester':
            self.transforms = nn.ModuleList([SylvesterTransform(dim) for i in range(num_transforms)])
        elif transform == 'Householder':
            self.transforms = nn.ModuleList([HouseholderTransform(dim) for i in range(num_transforms)])

        elif transform == 'RNVP':
            self.transforms = nn.ModuleList([RNVP(dim) for i in range(num_transforms)])

        elif transform == 'MNF':
            self.transforms = nn.ModuleList([MNF(dim) for i in range(num_transforms)])
        elif transform == 'mixed':  
            self.transforms = nn.ModuleList([HouseholderTransform(dim), PlanarTransform(dim),
                                             HouseholderTransform(dim), PlanarTransform(dim),
                                             HouseholderTransform(dim), PlanarTransform(dim),
                                             HouseholderTransform(dim), PlanarTransform(dim),
                                             HouseholderTransform(dim), PlanarTransform(dim),
                                             ])
        
        else:
            print('Transform not implemented')
    def forward(self, z):
        logdet = 0
        for f in self.transforms:
            z = f(z)
            logdet += f.log_det()
        return z,logdet

class RadialTransform(nn.Module):
    def __init__(self,dim):
        super().__init__() 
        self.z_0 = nn.Parameter(parameter_init(-0.1,0.1,dim))  
        self.log_alpha = nn.Parameter(parameter_init(-4,5,1)) 
        self.beta = nn.Parameter(parameter_init(-0.1,0.1,1)) 
        self.d = dim
        self.softplus = nn.Softplus()
    
    def forward(self, z):
        alpha = self.softplus(self.log_alpha)
        diff = z - self.z_0
       
        r = torch.norm(diff, dim=list(range(1, self.z_0.dim())))
        self.H1 = self.beta / (alpha + r)
        self.H2 = - self.beta*r*(alpha + r) ** (-2)
        z_new = z + self.H1 + self.H2
        return z_new

    def log_det(self):
        logdet = (self.d - 1) * torch.log(1 + self.H1) + torch.log(1 + self.H1 + self.H2)
        return logdet


class PlanarTransform(nn.Module):
    def __init__(self,dim):
        super().__init__() 
        #flow parameters
        self.u = nn.Parameter(parameter_init(-0.01,0.01,dim))  
        self.w = nn.Parameter(parameter_init(-0.01,0.01,dim)) 
        self.bias = nn.Parameter(parameter_init(-0.01,0.01,1)) 
        
    def h(self,x): #tanh activation function
        return torch.tanh(x)
    
    def h_derivative(self,x):
        return 1 - torch.tanh(x)**2
    
    def forward(self,z):
        inner = torch.dot(self.w,z) + self.bias
        z_new = z + self.u * (self.h(inner))
        self.psi =self.h_derivative(inner) * self.w 
        
        return z_new
        
    
    def log_det(self):
        return torch.log(torch.abs(1 + torch.dot(self.u,self.psi))) 
        

class SylvesterTransform(nn.Module):
    def __init__(self,dim):
        super().__init__() 
        self.M = 5
        self.A = nn.Parameter(parameter_init(-0.01,0.01,(dim,self.M)))  
        self.B = nn.Parameter(parameter_init(-0.01,0.01,(self.M,dim))) 
        self.b = nn.Parameter(parameter_init(-0.01,0.01,self.M)) 
    
    def h(self,x): #tanh activation function
        return torch.tanh(x)
    
    def h_derivative(self,x):
        return 1 - torch.tanh(x)**2
    
    def forward(self,z):
        self.linear = torch.matmul(self.B,z) + self.b
        return z + torch.matmul(self.A,self.h(self.linear))
    
    def log_det(self):
        I = torch.diag(torch.ones(self.M,device = DEVICE))
        diag = torch.diag(self.h_derivative(self.linear.flatten()))
        BA = torch.matmul(self.B,self.A)
        return torch.log(torch.det(I +torch.matmul(diag,BA)))

class HouseholderTransform(nn.Module):

    def __init__(self,dim):
        super().__init__()
        self.v = nn.Parameter(parameter_init(-0.01,0.01,dim))
       
    def forward(self,z):
        vtz = torch.dot(self.v,z)
        vvtz = self.v * vtz
        z_new = z - 2*vvtz/torch.sum(self.v**2)
        return z_new   
    
    def log_det(self):
        return 0


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.round()  # convert to (0,1)

    @staticmethod
    def backward(ctx, grad_output):
        # return F.hardtanh(grad_output)
        return grad_output


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class NN(nn.Module):
    """
    Simple fully connected neural network.
    """

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),
            StraightThroughEstimator()  # gradient of rounding operation is identity
        )

    def forward(self, x):
        return self.network(x)

class MLP(nn.Sequential):
    """Multilayer perceptron"""

    def __init__(self, *layer_sizes, leaky_a=0.1):
        layers = []
        for s1, s2 in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Linear(s1, s2))
          #  layers.append(nn.BatchNorm1d(s2))
            layers.append(nn.LeakyReLU(leaky_a))
        super().__init__(*layers[:-1])  # drop last ReLU


class RNVP(nn.Module):
    """Affine half flow aka Real Non-Volume Preserving (x = z * exp(s) + t),
    where a randomly selected half z1 of the dimensions in z are transformed as an
    affine function of the other half z2, i.e. scaled by s(z2) and shifted by t(z2).
    From "Density estimation using Real NVP", Dinh et al. (May 2016)
    https://arxiv.org/abs/1605.08803
    This implementation uses the numerically stable updates introduced by IAF:
    https://arxiv.org/abs/1606.04934
    """

    def __init__(self, dim, h_sizes=[75,75,75,75]):
        super().__init__()
        self.network = MLP(*[dim] + h_sizes)
        self.t = nn.Linear(h_sizes[-1], dim)
        self.s = nn.Linear(h_sizes[-1], dim)
        self.eps = 1e-6
        self.gate = 0
        self.mask = 0
    def forward(self, z):  # z -> x
        # Get random Bernoulli mask. This decides which channels will remain
        # unchanged and which will be transformed as functions of the unchanged.
        self.mask = torch.bernoulli(0.5 * torch.ones_like(z))

        z1, z2 = (1 - self.mask) * z, self.mask * z
        y = self.network(z2)
        shift, scale = self.t(y), self.s(y)
        self.gate = torch.sigmoid(scale)
        x = (z1 * self.gate + (1 - self.gate) * shift) + z2
        return x

    def log_det(self):
        return ((1 - self.mask) * self.gate.log()).sum(-1)





class MNF(nn.Module):
    def __init__(self, dim, hidden=100):
        super().__init__()
        self.f = nn.Linear(dim, hidden)
        self.g = nn.Linear(hidden, dim)
        self.k = nn.Linear(hidden, dim)
        self.bern = torch.distributions.Bernoulli(probs=.5)

    def forward(self, z):
        self.m = torch.bernoulli(0.5 * torch.ones_like(z))
        h = torch.tanh(self.f(self.m * z))
        mu = self.g(h)
        self.sigma = torch.sigmoid(self.k(h))
        return self.m * z + (1 - self.m) * (z * self.sigma + (1 - self.sigma) * mu)

    def log_det(self):
        return ((1 - self.m) * self.sigma.log()).sum()
