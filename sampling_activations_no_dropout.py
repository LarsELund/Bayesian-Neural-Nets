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
#torch.manual_seed(1)

"""Uses variational inference with the local reparametrization trick to sample the activations
(without variational dropout) of a fully connected neural network with 3 hidden layers. 
KL divergence is computed analytically between normal priors and posteriors.
"""


root = './'
config = {
          'num_classes':10,  
          'batch_size': 100,
          'use_cuda': False,      #True=use Nvidia GPU | False use CPU
          'log_interval': 100,     #How often to display (batch) loss during training
          'epochs':5,    
          'num_test_ensemble_samples': 10, #wisdom of the crowds..
          'lr': 1e-4
         }

if config['use_cuda']:
    device = 'cuda'
else:
    device = 'cpu'


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root, train=True, download= False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=config['batch_size'], shuffle=False)

val_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root, train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=config['batch_size'], shuffle=False)



def parameter_init(low,high,size):
    #used to initialize the parameters, random uniform
    random_init = (low - high) * torch.rand(size = size,device = device) + high
    return random_init  

def KL_divergence(mu1,sigma1,mu2,sigma2):
    #KL between two normals can be computed analytically
    KL = torch.sum(torch.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2)/(2*sigma2**2) - 0.5)
    return KL



class BayesianLayer(nn.Module):
    def __init__(self,n,m): #n is size of input, m output from layer
        super().__init__()
        self.softplus = nn.Softplus()
        self.mu = nn.Parameter(parameter_init(-0.01,0.01,(n,m)))
        self.rho = nn.Parameter(parameter_init(-5,-4,(n,m)))
        self.prior_mu = torch.zeros((n,m),device = device) #prior is N(0,1)
        self.prior_sigma = torch.zeros((n,m),device = device) + 1
        
        
    
    def forward(self,x):
        self.sigma = self.softplus(self.rho) #make sure that sigma is always positive
        self.phi = torch.matmul(x,self.mu) #these are eq(6) from Kingma et al 
        self.delta = torch.matmul(x**2,self.sigma**2)  
        zeta = torch.randn(size = (self.phi.size()),device = device) #sample N(0,1)
        activations = self.phi + torch.sqrt(self.delta) * zeta #reparametrization trick
        return activations
    
            

class BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BayesianLayer(28*28,1200)
        self.l2 = BayesianLayer(1200,1200)
        self.l3 = BayesianLayer(1200,1200)
        self.l4 = BayesianLayer(1200,10)
       
    def forward(self,x):
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
     KL = 0
     for layer in model.children(): #sum up the KL divergence for all layers
         div = KL_divergence(layer.mu,layer.sigma,layer.prior_mu,layer.prior_sigma)
         KL += div
     negative_log_lik = F.nll_loss(prediction, target,reduction = 'sum') 
     loss = KL/num_batches + negative_log_lik
     return loss


model = BNN()
model.to(device)
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
    
    if is_training: 
        model.train()
        
    else:
        model.eval()
        
    total_loss       = 0 
    correct          = 0 
    confusion_matrix = np.zeros(shape=(10,10))
    labels_list      = [0,1,2,3,4,5,6,7,8,9]
    for batch_idx, data_batch in enumerate(data_loader):
        if config['use_cuda']:
            images = data_batch[0].to(device) # send data to GPU
            labels = data_batch[1].to(device) # send data to GPU
        else:
            images = data_batch[0]
            labels = data_batch[1]
            
        if not is_training:
           
            with torch.no_grad():
                i,j,k = config['num_test_ensemble_samples'],data_batch[1].shape[0],config['num_classes']
                all_predictions = torch.zeros((i,j,k),device = device)
                
                for x in range(i): #model ensemble
                    all_predictions[x,:,:] = model(images)
               
                prediction = all_predictions.mean(dim = 0)
                loss = loss_fn(prediction,labels,model)
                total_loss += loss.cpu().detach().numpy()
        
        elif is_training:
           
            model.zero_grad()
            prediction = model(images)
            loss = loss_fn(prediction,labels,model)
            total_loss += loss.cpu().detach().numpy()
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
                                   
