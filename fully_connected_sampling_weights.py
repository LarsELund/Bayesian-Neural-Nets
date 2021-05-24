#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

"""Uses variational inference to sample the weights and biases of 
a fully connected bayesian neural network with two hidden layers. the priors 
for the different layers can be gaussian or laplace, the variational
posterior is gaussian. Gives similar results as when sampling the 
activations instead of the weights."""


root = './' #data exists in working directory

config = {
          'num_classes':10,  
          'batch_size': 100,
          'use_cuda': False,      #True=use Nvidia GPU | False use CPU
          'log_interval': 100,     #How often to display (batch) loss during training
          'epochs':10,    
          'num_elbo_samples': 1,
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

class Laplace:
    def __init__(self,loc,scale):
        self.loc = loc
        self.scale = scale
        
    def sample(self): #sample the weights, using the reparametrization trick
        epsilon = torch.randn(self.loc.size(),device = device)
        W = self.loc + self.scale * epsilon
        return W
    
    def log_prob(self,w):
        laplace = torch.distributions.laplace.Laplace(self.loc, self.scale, validate_args=None)
        return laplace.log_prob(w).sum()

class Gaussian: 
    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho
        self.Softplus = nn.Softplus()
    
    @property
    def sigma(self):
        return self.Softplus(self.rho) #force sigma to be positive
    
    def sample(self): #sample the weights, using the reparametrization trick
        epsilon = torch.randn(self.rho.size(),device = device)
        W = self.mu + self.sigma * epsilon
        return W
    
    def log_prob(self, w):
        normal = torch.distributions.normal.Normal(self.mu,self.sigma, validate_args=None)
        log_prob = normal.log_prob(w).sum()
        return log_prob


class BayesianLayer(nn.Module):
    def __init__(self, in_features, out_features,prior):
        super().__init__()
        n = in_features
        m = out_features
        # intiaialize mu and rho randomly, for the weights
        
        random_init_weight_mu = torch.Tensor(size =(m,n)).uniform_(-0.01,0.01).to(device)
        random_init_weight_rho = torch.Tensor(size =(m,n)).uniform_(-5,-4).to(device)
    
        self.weight_mu = nn.Parameter(random_init_weight_mu)
        self.weight_rho = nn.Parameter(random_init_weight_rho)
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        
        # same for the bias
        
        random_init_bias_mu = torch.Tensor(m).uniform_(-0.01,0.01).to(device)
        random_init_bias_rho = torch.Tensor(m).uniform_(-5,-4).to(device)
        self.bias_mu = nn.Parameter(random_init_bias_mu)
        self.bias_rho = nn.Parameter(random_init_bias_rho)
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
       
        # Prior distributions, this gives N(0,1) for all W and bias
        prior_weight_loc = torch.zeros((m,n),device = device)
        prior_weight_scale = (torch.zeros((m,n),device = device) + 1)
        prior_bias_loc = torch.zeros(m,device = device)
        prior_bias_scale = (torch.zeros(m,device = device) + 1)
        
        if prior == 'Laplacian':
            self.weight_prior = Laplace(prior_weight_loc,prior_weight_scale)
            self.bias_prior = Laplace(prior_bias_loc,prior_bias_scale) 
        
        else:
            self.weight_prior = Gaussian(prior_weight_loc,prior_weight_scale) 
            self.bias_prior = Gaussian(prior_bias_loc,prior_bias_scale)
            
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self,x): 

        weight = self.weight.sample()
        bias = self.bias.sample()
        self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
        self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        return F.linear(x, weight, bias)


class BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BayesianLayer(28*28,400,'Laplacian')
        self.l2 = BayesianLayer(400,400,'Laplacian')
        self.l3 = BayesianLayer(400,10,'Gaussian')
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.log_softmax(self.l3(x), dim=1)
        return x
    
    def log_prior(self): 
        return sum([layer.log_prior for layer in self.children()])
        
    
    def log_variational_posterior(self):
        return sum([layer.log_variational_posterior for layer in self.children()])
    

def loss_fn(input,target,num_samples,model): #sample the ELBO
        M = config['batch_size']
        
        if model.train():
            N = len(train_loader.dataset)
        else:
            N = len(val_loader.dataset)
        
        num_batches = N/M 
        outputs = torch.zeros(num_samples, input.shape[0], config['num_classes'],device = device)
        log_priors = torch.zeros(num_samples)
        log_variational_posteriors = torch.zeros(num_samples)
        for i in range(num_samples): 
            outputs[i] = model(input) 
            log_priors[i] = model.log_prior()
            log_variational_posteriors[i] = model.log_variational_posterior()
        log_prior = log_priors.mean() 
        log_variational_posterior = log_variational_posteriors.mean()
        output = outputs.mean(dim = 0)
        negative_log_lik = F.nll_loss(output, target,reduction = 'sum') 
        loss = (log_variational_posterior - log_prior)/num_batches + num_batches*negative_log_lik
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
               loss = loss_fn(images,labels,config['num_elbo_samples'],model)
               total_loss += loss.cpu().detach().numpy()
        elif is_training:
            
            prediction = model(images)
            loss = loss_fn(images,labels,config['num_elbo_samples'],model)
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
    run_model()
    do_plotting()
    find_acc()                              
                                   