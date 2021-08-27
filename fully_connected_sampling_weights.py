#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import math
import time

"""Samples the weights and biases of a 3 layer fully connected neural network.
Priors are laplace and posteriors are gaussian. the KL divergence is approximated
by sampling the ELBO (once)."""


root = './' 

config = {
          'num_classes':10,  
          'batch_size': 100,
          'use_cuda': False,      #True=use Nvidia GPU | False use CPU
          'log_interval': 100,     #How often to display (batch) loss during training
          'epochs':20,    
          'num_test_ensemble_samples': 10,
          'lr': 0.001
         }


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root, train=True, download= False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=config['batch_size'], shuffle=True)

val_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root, train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=config['batch_size'], shuffle=False)


if config['use_cuda']:
    device = 'cuda'
else:
    device = 'cpu'
    
def normal_log_prob(mu,sigma,w):
    return torch.sum(-torch.log(sigma)-0.5*torch.log(torch.tensor(2*math.pi))-(w-mu)**2 /(2*sigma**2))

def laplace_log_prob(loc,scale,w):
    return torch.sum(-torch.log(2*scale) - torch.abs(w - loc)/scale)

def parameter_init(low,high,size):
    #used to initialize the parameters, random uniform
    random_init = (low - high) * torch.rand(size,device = device) + high
    return random_init  



class BayesianLayer(nn.Module):
    def __init__(self,n,m):
        super().__init__() 
        self.softplus = nn.Softplus()
        self.w_mu = nn.Parameter(parameter_init(-0.01,0.01,(m,n))) 
        self.w_rho = nn.Parameter(parameter_init(-5,-4,(m,n)))
        self.b_mu = nn.Parameter(parameter_init(-0.01,0.01,m))
        self.b_rho = nn.Parameter(parameter_init(-5,-4,m))
      
        # prior is laplace(0,1) for all w and b
        self.prior_w_loc = torch.zeros((m,n),device = device) #weight prior
        self.prior_w_scale = (torch.zeros((m,n),device = device) + 1)
        self.prior_b_loc = torch.zeros(m,device = device) #bias prior
        self.prior_b_scale = (torch.zeros(m,device = device) + 1)
        

    def forward(self,x): 
        self.w_sigma = self.softplus(self.w_rho) #make sure sigma is positive
        self.b_sigma = self.softplus(self.b_rho)
        e1 = torch.randn(self.w_rho.size(),device = device) #one for the weights
        e2 = torch.randn(self.b_rho.size(),device = device) #one for the biases
        self.Weight = self.w_mu + self.w_sigma * e1 #reparametrization trick
        self.Bias = self.b_mu + self.b_sigma * e2
        return F.linear(x,self.Weight, self.Bias)


class BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BayesianLayer(28*28,1200)
        self.l2 = BayesianLayer(1200,1200)
        self.l3 = BayesianLayer(1200,1200)
        self.l4 = BayesianLayer(1200,10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.log_softmax(self.l4(x), dim=1)
        return x
    
def loss_fn(prediction,target,model):
     M = config['batch_size']
     if model.train():
         N = len(train_loader.dataset)
     else:
         N = len(val_loader.dataset)    
     num_batches = N/M 
     log_posterior = 0 
     log_prior = 0
     for layer in model.children(): #just sum the log probs for each layer for the weights and biases
         log_posterior += normal_log_prob(layer.w_mu,layer.w_sigma,layer.Weight)
         log_posterior += normal_log_prob(layer.b_mu,layer.b_sigma,layer.Bias)
         log_prior += laplace_log_prob(layer.prior_w_loc,layer.prior_w_scale,layer.Weight)
         log_prior += laplace_log_prob(layer.prior_b_loc,layer.prior_b_scale,layer.Bias)
     
     negative_log_lik = F.nll_loss(prediction, target,reduction = 'sum') 
     loss = (log_posterior - log_prior)/num_batches + negative_log_lik
     return loss
    
model = BNN()
if config['use_cuda']:
    model.to('cuda')
optimizer = torch.optim.AdamW(model.parameters(),lr = config['lr'])

def run_epoch(model, epoch, data_loader, optimizer, is_training, config):
    """
    Args:
        model        (obj): The neural network model
        epoch        (int): The current epoch
        data_loader  (obj): A pytorch data loader "torch.utils.data.DataLoader"
        optimizer    (obj): A pytorch optimizer "torch.optim"
        is_training (bool): Whether to use train (update) the model/weights or not. 
        config      (dict): Configuration parameters

    Intermediate:
        totalLoss: (float): The accumulated loss from all batches. 
                            Hint: Should be a numpy scalar and not a pytorch scalar

    Returns:
        loss_avg         (float): The average loss of the dataset
        accuracy         (float): The average accuracy of the dataset
        confusion_matrix (float): A 10x10 matrix
    """
    
    if is_training == True: 
        model.train()
        
    else:
        model.eval()
        
       
    total_loss       = 0 
    correct          = 0 
    confusion_matrix = np.zeros(shape=(10,10))
    labels_list      = [0,1,2,3,4,5,6,7,8,9]
    for batch_idx, data_batch in enumerate(data_loader):
        if config['use_cuda'] == True:
            images = data_batch[0].to(device) # send data to GPU
            labels = data_batch[1].to(device) # send data to GPU
        else:
            images = data_batch[0]
            labels = data_batch[1]
            
        if not is_training:
            with torch.no_grad():
                
               i,j,k = config['num_test_ensemble_samples'],data_batch[1].shape[0],config['num_classes']
               all_predictions = torch.zeros((i,j,k),device = device)
               for x in range(i):
                   all_predictions[x] = model(images)
               prediction = all_predictions.mean(dim = 0)
               loss = loss_fn(prediction,labels,model)
               total_loss += loss.cpu().detach().numpy()
        elif is_training:
            
            prediction = model(images)
            loss = loss_fn(prediction,labels,model)
            total_loss += loss.cpu().detach().numpy()
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            
            
       
            
     

        # Update the number of correct classifications and the confusion matrix
        predicted_label  = prediction.max(1, keepdim=True)[1][:,0]
        correct          += predicted_label.eq(labels).cpu().sum().numpy()
        confusion_matrix += metrics.confusion_matrix(labels.cpu().numpy(), predicted_label.cpu().numpy(), labels=labels_list)


        # Print statistics
        #batchSize = len(labels)
        if batch_idx % config['log_interval'] == 0:
            print(f'Epoch={epoch} | {(batch_idx+1)/len(data_loader)*100:.2f}% | loss = {loss:.5f}')

    loss_avg         = total_loss / len(data_loader)
    accuracy         = correct / len(data_loader.dataset)
    confusion_matrix = confusion_matrix / len(data_loader.dataset)

    return loss_avg, accuracy, confusion_matrix


train_loss = np.zeros(shape=config['epochs'])
train_acc  = np.zeros(shape=config['epochs'])
val_loss   = np.zeros(shape=config['epochs'])
val_acc    = np.zeros(shape=config['epochs'])
train_confusion_matrix = np.zeros(shape=(10,10,config['epochs']))
val_confusion_matrix   = np.zeros(shape=(10,10,config['epochs']))

def run_model():
    for epoch in range(config['epochs']):
       
       train_loss[epoch], train_acc[epoch], train_confusion_matrix[:,:,epoch] = \
                                   run_epoch(model, epoch, train_loader, optimizer, is_training=True, config=config)
       val_loss[epoch], val_acc[epoch], val_confusion_matrix[:,:,epoch]     = \
                                 run_epoch(model, epoch, val_loader, optimizer, is_training=False, config=config)
                                   
                                   
def do_plotting():
    
    plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    ax = plt.subplot(2, 1, 1)
    # plt.subplots_adjust(hspace=2)
    ax.plot(train_loss, 'b', label='train loss')
    ax.plot(val_loss, 'r', label='validation loss')
    ax.grid()
    plt.ylabel('Loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    ax.legend(loc='upper right', fontsize=16)
    
    ax = plt.subplot(2, 1, 2)
    plt.subplots_adjust(hspace=0.4)
    ax.plot(train_acc, 'b', label='train accuracy')
    ax.plot(val_acc, 'r', label='validation accuracy')
    ax.grid()
    plt.ylabel('Accuracy', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    val_acc_max = np.max(val_acc)
    val_acc_max_ind = np.argmax(val_acc)
    plt.axvline(x=val_acc_max_ind, color='g', linestyle='--', label='Highest validation accuracy')
    plt.title('Highest validation accuracy = %0.1f %%' % (val_acc_max*100), fontsize=16)
    ax.legend(loc='lower right', fontsize=16)
    plt.ion()
    plt.savefig('plot.png')

def find_acc():
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ind = np.argmax(val_acc)
    class_accuracy = val_confusion_matrix[:,:,ind]
    for ii in range(len(classes)):
        acc = val_confusion_matrix[ii,ii,ind] / np.sum(val_confusion_matrix[ii,:,ind])
        print(f'Accuracy of {str(classes[ii]).ljust(15)}: {acc*100:.01f}%')

if __name__ == '__main__':
    t1 = time.time()
    run_model()
    do_plotting()
    find_acc()   
    time = round((time.time() - t1),1)  
    print('Took',time/config['epochs'],'seconds/epoch on',device)                
