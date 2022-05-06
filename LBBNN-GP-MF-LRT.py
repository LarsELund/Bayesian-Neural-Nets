#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

# define the summary writer
writer = SummaryWriter()
sns.set()
sns.set_style("dark")
sns.set_palette("muted")
sns.set_color_codes("muted")

# select the device
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# define the parameters
BATCH_SIZE = 100
TEST_BATCH_SIZE = 1000
COND_OPT = False
CLASSES = 10
SAMPLES = 1
TEST_SAMPLES = 10
epochs = 250

# define the data loaders
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        './mnist', train=True, download=True,
        transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **LOADER_KWARGS)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        './mnist', train=False, download=True,
        transform=transforms.ToTensor()),
    batch_size=TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)


### the loaders used for computing the OOD entropy
fmnist_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./fmnist', train=False, download=True,
    transform=transforms.ToTensor()), batch_size=TEST_BATCH_SIZE, shuffle=False)

kmnist_loader = torch.utils.data.DataLoader(
    datasets.KMNIST('./kmnist', train=False, download=True,
    transform=transforms.ToTensor()), batch_size=TEST_BATCH_SIZE, shuffle=False)

TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(test_loader.dataset)
NUM_BATCHES = len(train_loader)
NUM_TEST_BATCHES = len(test_loader)

assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0

# define Gaussian distribution
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def rsample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

    def log_prob_iid(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2))

    def full_log_prob(self, input, gamma):
        return (torch.log(gamma * (torch.exp(self.log_prob_iid(input)))
                          + (1 - gamma) + 1e-8)).sum()


# define Bernoulli distribution
class Bernoulli(object):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.exact = True

    def rsample(self):
        if self.exact:
            gamma = torch.distributions.Bernoulli(self.alpha).sample().to(DEVICE)
        else:
            gamma = torch.distributions.RelaxedBernoulli(probs=self.alpha, temperature=TEMPER_PRIOR).rsample()
        # Variable((self.bern.sample(alpha.size()).to(DEVICE)<alpha.type(torch.cuda.FloatTensor)).type(torch.cuda.FloatTensor)).to(DEVICE)
        return gamma

    def sample(self):
        self.bern.sample()

    def log_prob(self, input):
        if self.exact:
            gamma = torch.round(input.detach())
            output = (gamma * torch.log(self.alpha + 1e-8) + (1 - gamma) * torch.log(1 - self.alpha + 1e-8)).sum()
        else:
            output = (input * torch.log(self.alpha + 1e-8) + (1 - input) * torch.log(1 - self.alpha + 1e-8)).sum()
        return output

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # weight variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # weight prior is iid N(0,1)
        self.mu_prior = torch.zeros(out_features,in_features,device = DEVICE)
        self.sigma_prior = (self.mu_prior + 1.).to(DEVICE)

        # model variational parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(0,1))
        self.alpha_q = torch.Tensor(out_features, in_features).uniform_(0.999, 0.9999)
        self.gamma = Bernoulli(self.alpha_q)

        # inclusion prior is iid Bernoulli(0.05)
        self.alpha_prior = (self.mu_prior + 0.05).to(DEVICE)

        # bias variational parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # bias prior isi iid N(0,1)
        self.bias_mu_prior = torch.zeros(out_features,device = DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + 1.).to(DEVICE)

        # scalars
        self.kl = 0

    # forward path
    def forward(self, input, sample=False, calculate_log_probs=False):
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))
        self.gamma.alpha = self.alpha_q
        if self.training or sample:
            e_w = self.weight.mu * self.alpha_q
            var_w = self.weight.sigma**2 * self.alpha_q ** 2
            e_b = torch.mm(input,e_w.T) + self.bias.mu
            var_b = torch.mm(input **2,var_w.T) + self.bias.sigma**2
            eps = torch.randn(size = (e_b.size()),device = DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps # eq (7.4)

        else:
            e_w = self.weight.mu * self.alpha_q
            e_b = torch.mm(input,e_w.T) + self.bias.mu
            activations = e_b

        if self.training or calculate_log_probs:

            # eq 7.4
            kl_bias = (torch.log(self.bias_sigma_prior / self.bias.sigma) - 0.5 + (self.bias.sigma**2
            + (self.bias.mu-self.bias_mu_prior)**2) / (2*self.bias_sigma_prior**2)).sum()

            # eq 7.3
            kl_weight = (self.alpha_q * (torch.log(self.sigma_prior / self.weight.sigma)
                        - 0.5 + torch.log(self.alpha_q / self.alpha_prior)
                        + (self.weight.sigma**2 + (self.weight.mu-self.mu_prior)**2)/(2*self.sigma_prior**2))
                        + (1 - self.alpha_q) * torch.log((1 - self.alpha_q)/(1-self.alpha_prior))).sum()

            self.kl = kl_bias + kl_weight
        else:
            self.kl = 0
        return activations

class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__() # define the architecture of the network, same as in baseline method
        self.l1 = BayesianLinear(28 * 28, 400)
        self.l2 = BayesianLinear(400, 600)
        self.l3 = BayesianLinear(600, 10)

    def forward(self, x, sample=False):# forward propagation from 28x28 img input to softmax output
        x = x.view(-1, 28*28)
        x = F.relu(self.l1.forward(x,sample))
        x = F.relu(self.l2.forward(x,sample))
        x = F.log_softmax((self.l3.forward(x, sample)), dim=1)
        return x
    
    def kl(self): # sum up kl divergence for the three layers
        return self.l1.kl + self.l2.kl + self.l3.kl

# Stochastic Variational Inference iteration
def train(net, optimizer):
    net.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        outputs = net(data, sample=True)
        negative_log_likelihood = F.nll_loss(outputs, target, reduction='sum') #compute nll between y_hat and y
        ELBO = negative_log_likelihood + net.kl() / NUM_BATCHES #the approximate negative ELBO (one sample)
        ELBO.backward()
        optimizer.step()
    print('ELBO',ELBO.item())
    print('nll',negative_log_likelihood.item())
    return negative_log_likelihood.item(), ELBO.item()

def test_ensemble(net):
    net.eval()
    metr = []
    density = np.zeros(TEST_SAMPLES)
    ensemble = []
    posterior_mean = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                g1 = net.l1.gamma.rsample().to(DEVICE)
                g2 = net.l2.gamma.rsample().to(DEVICE)
                g3 = net.l3.gamma.rsample().to(DEVICE)
                gamms = torch.cat((g1.flatten(), g2.flatten(), g3.flatten()))
                density[i] = gamms.mean()
                outputs[i] =net(data, sample=True)
                if (i == 0):
                    mydata_means = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        mydata_means[j] /= np.sum(mydata_means[j])
                else:
                    tmp = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        tmp[j] /= np.sum(tmp[j])
                    # print(sum(tmp[j]))
                    mydata_means = mydata_means + tmp

            mydata_means /= TEST_SAMPLES
            np.savetxt('mnist_means_act' + '.txt', mydata_means, delimiter=',')
            d = data.reshape(1000, 28 * 28).cpu().numpy()
            np.savetxt('mnist_data_act' + '.txt', d, delimiter=',')
            output1 = outputs[0:10].mean(0) #ensemble
            pred1 = output1.max(1, keepdim=True)[1]  # index of max log-probability
            mean_out = net(data, sample=False) #posterior mean
            pr = mean_out.max(1, keepdim=True)[1]
            p = pred1.squeeze().cpu().numpy()
            a = pr.eq(target.view_as(pred1)).sum(dim=1).squeeze().cpu().numpy()
            b = pred1.eq(target.view_as(pred1)).sum().item()
            posterior_mean.append(a.sum())
            ensemble.append(b)
            np.savetxt('mnist_predictions_act' + '.txt', p, delimiter=',')
            np.savetxt('mnist_truth_act' + '.txt', target.cpu().numpy(), delimiter=',')
    metr.append(np.sum(posterior_mean) / TEST_SIZE)
    metr.append(np.sum(ensemble) / TEST_SIZE)
    print(np.mean(density), 'density')
    metr.append(np.mean(density))
    print(np.sum(posterior_mean) / TEST_SIZE,'posterior mean')
    print(np.sum(ensemble) / TEST_SIZE, 'ensemble')
    return metr


def cdf(x, plot=True, *args, **kwargs):
    x = sorted(x)
    y = np.arange(len(x)) / len(x)
    return plt.plot(x, y, *args, **kwargs) if plot else (x, y)


from scipy.special import expit

def sigmoid(x):
    return expit(x)


def outofsample(net,loader,medimod = False):
    net.eval()
    correct = 0
    corrects = np.zeros(TEST_SAMPLES + 2, dtype=int)
    entropies = np.zeros(10)
    count = 0
    k = 0
    spars = np.zeros(TEST_SAMPLES)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES + 2, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                # print(i)
                if medimod:
                    outputs[i] = net.forward(data, sample=True,
                                             g1=(net.l1.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                             g2=(net.l2.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE),
                                             g3=(net.l3.alpha.data > 0.5).type(torch.cuda.FloatTensor).to(DEVICE))
                else:
                    outputs[i] = net(data, sample=True)


                if (i == 0):
                    mydata_means = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        mydata_means[j] /= np.sum(mydata_means[j])
                else:
                    tmp = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        tmp[j] /= np.sum(tmp[j])
                    # print(sum(tmp[j]))
                    mydata_means = mydata_means + tmp
            mydata_means /= TEST_SAMPLES
            # print(np.sum(mydata_means))
            for j in range(TEST_BATCH_SIZE):
                if k == 0 and j == 0:
                    entropies = -np.sum(mydata_means[j] * np.log(mydata_means[j]))
                else:
                    entropies = np.append(entropies, -np.sum(mydata_means[j] * np.log(mydata_means[j])))
            k += 1
            output = outputs[1:TEST_SAMPLES].mean(0)
            preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1]  # index of max log-probability
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
    #cdf(entropies.flatten())
    return entropies.flatten()



import time
print("Classes loaded")

nll_several_runs = []
loss_several_runs = []
metrics_several_runs = []

# make inference on 10 networks
for i in range(0, 10):
    print('network',i)
    torch.manual_seed(i)
    net = BayesianNetwork().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    all_nll = []
    all_loss = []
    t1 = time.time()
    for epoch in range(epochs):
        print('epoch',epoch)
        nll,loss = train(net, optimizer)
        all_nll.append(nll)
        all_loss.append(loss)
    nll_several_runs.append(all_nll)
    loss_several_runs.append(all_loss)
    t = round((time.time() - t1), 1)
    metrics = test_ensemble(net)
    metrics.append(t / epochs)
    metrics_several_runs.append(metrics)
    if i == 9:
        enr = outofsample(net, loader=fmnist_loader)
        enr2 = outofsample(net, loader=kmnist_loader)
        print(enr.mean(), 'entr m - > f')
        print(enr2.mean(), 'entr m - > k')
        np.savetxt('ENTROPY_MNIST_FMNIST_ACT' + '.txt', enr, delimiter=',')
        np.savetxt('ENTROPY_MNIST_KMNIST_ACT' + '.txt', enr2, delimiter=',')

np.savetxt('MNIST_ACT_LOSS' + '.txt', loss_several_runs, delimiter=',',fmt='%s')
np.savetxt('MNIST_ACT_METRICS' + '.txt', metrics_several_runs, delimiter=',',fmt='%s')
np.savetxt('MNIST_ACT_NLL' + '.txt', nll_several_runs, delimiter=',',fmt='%s')
