#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def parameter_init(low,high,size):
    #used to initialize the parameters, random uniform
    random_init = (low - high) * torch.rand(size,device = DEVICE) + high
    return random_init  

class PropagateFlow(nn.Module):
    def __init__(self,transform,dim,num_transforms):
        super().__init__() 
        if transform == 'Planar':
            self.transforms = nn.ModuleList([PlanarTransform(dim) for i in range(num_transforms)])
        elif transform == 'Radial':
            self.transforms = nn.ModuleList([RadialTransform(dim) for i in range(num_transforms)])
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
        self.z_0 = nn.Parameter(parameter_init(-0.01,0.01,dim))  
        self.log_alpha = nn.Parameter(parameter_init(-0.01,0.01,1)) 
        self.beta = nn.Parameter(parameter_init(-5,5,1)) 
        self.d = dim
        self.softplus = nn.Softplus()
    
    def forward(self, z):
        alpha = self.softplus(self.log_alpha) #make sure alpha is always positive
        diff = z - self.z_0
        r = torch.norm(diff)
        self.H1 = self.beta / (alpha + r)
        self.H2 = - self.beta*r*(alpha + r) ** (-2)
        z_new = z + self.H1 + self.H2
        return z_new

    def log_det(self):
        logdet = (self.d - 1) * torch.log(1 + self.H1) + torch.log(1 + self.H1 + self.H2)
        return logdet[0]


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
        
        
    
    