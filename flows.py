#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

config = {
          'num_classes':10,  
          'batch_size': 100,
          'use_cuda': False,      #True=use Nvidia GPU | False use CPU
          'log_interval': 100,     #How often to display (batch) loss during training
          'epochs': 50,    
          'num_test_ensemble_samples': 10,
          'lr': 1e-4
         }

if config['use_cuda']:
    device = 'cuda'
else:
    device = 'cpu'
    


def parameter_init(low,high,size):
    #used to initialize the parameters, random uniform
    random_init = (low - high) * torch.rand(size,device = device) + high
    return random_init  

class PropagateFlows(nn.Module):
    def __init__(self,flow,dim,num_flows):
        super().__init__() 
        if flow == 'PlanarFlow':
            self.flow = nn.ModuleList([PlanarFlow(dim) for i in range(num_flows)])
        elif flow == 'RadialFlow':
            self.flow = nn.ModuleList([RadialFlow(dim) for i in range(num_flows)])
        else:
            print('Flow not implemented')
    def forward(self, z):
        logdet = 0
        for f in self.flow:
            z = f(z)
            logdet += f.log_det()
        return z,logdet

    

class RadialFlow(nn.Module):
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


class PlanarFlow(nn.Module):
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
        
        
    
    