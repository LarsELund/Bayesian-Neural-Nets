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

def propagate_flows(z,flow):
    logdet = 0
    for f in flow:
        z,ld = f(z)
        logdet += ld
    return z, logdet


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
        h1 = self.beta / (alpha + r)
        h2 = - self.beta*r*(alpha + r) ** (-2)
        z_new = z + h1 + h2
        log_det = (self.d - 1) * torch.log(1 + h1) + torch.log(1 + h1 + h2)
        return z_new, log_det[0]



class PlanarFlow(nn.Module):
    def __init__(self,dim):
        super().__init__() 
        
        #initialize flow parameters
        #each layer has n*m weights and m bias parameters
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
        psi =self.h_derivative(inner) * self.w 
        logdet = torch.log(torch.abs(1 + torch.dot(self.u,psi)))
        return z_new,logdet