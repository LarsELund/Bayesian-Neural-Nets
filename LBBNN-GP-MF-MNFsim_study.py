#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# matplotlib inline
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import pandas as pd
from flows_simstudy import PropagateFlow



np.random.seed(1)

writer = SummaryWriter()
sns.set()
sns.set_style("dark")
sns.set_palette("muted")
sns.set_color_codes("muted")

# select the device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

Z_FLOW_TYPE = 'RNVP'
R_FLOW_TYPE = 'RNVP'

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# define the parameters
BATCH_SIZE = 400
TEST_BATCH_SIZE = 10
COND_OPT = False
CLASSES = 2
SAMPLES = 1
TEST_SAMPLES = 10
epochs = 500



# import the data

x_df = pd.read_csv('sim3-X.csv', header=None)
y_df = pd.read_csv('sim3-Y.csv', header=None)
x = np.array(x_df)
y = np.array(y_df)
data = torch.tensor(np.column_stack((x, y)), dtype=torch.float32)

# tr_ids = np.random.choice(2000, 1600, replace = False) ## 1600 train data, 400 test data
# te_ids = np.setdiff1d(np.arange(2000),tr_ids)

# dtrain = data[tr_ids,:]


data_mean = data.mean(axis=0)[6:20]
data_std = data.std(axis=0)[6:20]

data[:, 6:20] = (data[:, 6:20] - data_mean) / data_std

dtrain = data
# dtest = data[te_ids,:]
TRAIN_SIZE = len(dtrain)
# TEST_SIZE = len(te_ids)
NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE
# NUM_TEST_BATCHES = len(te_ids)/BATCH_SIZE


n, p = dtrain.shape

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
        # return torch.exp(self.rho)

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


# define low rank multivariate Gaussian distribution
class LowRankMultivariateNormal(torch.distributions.MultivariateNormal):
    pass
    # rsample, log_prob, etc. available by inheritance


# define Bernoulli distribution
class Bernoulli(object):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.exact = False

    def rsample(self):
        if self.exact:
            gamma = torch.distributions.Bernoulli(self.alpha).sample().to(DEVICE)
        else:
            gamma = torch.distributions.RelaxedBernoulli(probs=self.alpha, temperature=TEMPER_PRIOR).rsample()
        return gamma

    def sample(self):
        return torch.distributions.Bernoulli(self.alpha).sample().to(DEVICE)

    def log_prob(self, input):
        if self.exact:
            gamma = torch.round(input.detach())
            output = (gamma * torch.log(self.alpha + 1e-8) + (1 - gamma) * torch.log(1 - self.alpha + 1e-8)).sum()
        else:
            output = (input * torch.log(self.alpha + 1e-8) + (1 - input) * torch.log(1 - self.alpha + 1e-8)).sum()
        return output


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, num_transforms):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.01, 0.01))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # weight priors
        self.mu_prior = torch.zeros(out_features, in_features, device=DEVICE) + 0.1
        self.sigma_prior = (self.mu_prior + 1.2).to(DEVICE)

        # model parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(1.5, 2.5))
        self.alpha_q = torch.Tensor(out_features, in_features).uniform_(0.999, 0.9999)
        self.gamma = Bernoulli(self.alpha_q)

        # model priors
        self.alpha_prior = (self.mu_prior + 0.2).to(DEVICE)
        # self.gamma_prior.exact = True
        # bias (intercept) parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # bias (intercept) priors
        self.bias_mu_prior = torch.zeros(out_features, device=DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + 1.3).to(DEVICE)

        self.q0_mean = nn.Parameter(0.1 * torch.randn(in_features))
        self.q0_log_var = nn.Parameter(-9 + 0.1 * torch.randn(in_features))
        # auxiliary variable c, b1 and b2 are defined in equation (9) and (10)
        self.r0_c = nn.Parameter(0.1 * torch.randn(in_features))
        self.r0_b1 = nn.Parameter(0.1 * torch.randn(in_features))
        self.r0_b2 = nn.Parameter(0.1 * torch.randn(in_features))
        self.z_flow = PropagateFlow(Z_FLOW_TYPE, in_features, num_transforms)
        self.r_flow = PropagateFlow(R_FLOW_TYPE, in_features, num_transforms)

        self.kl = 0
        self.z = 0

    def sample_z(self, batch_size=1):
        q0_std = self.q0_log_var.exp().sqrt().repeat(batch_size, 1)
        epsilon_z = torch.randn_like(q0_std)
        self.z = self.q0_mean + q0_std * epsilon_z
        zs, log_det_q = self.z_flow(self.z)
        return zs[-1], log_det_q.squeeze()

        # forward path

    def forward(self, input, sample=False, calculate_log_probs=False):
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))
        self.gamma.alpha = self.alpha_q
        if self.training or sample:
            z_k, _ = self.sample_z(input.size(0))
            e_w = self.weight.mu * self.alpha_q

            var_w = self.weight.sigma **2 * self.alpha_q**2
            e_b = torch.mm(input * z_k, e_w.T) + self.bias.mu
            var_b = torch.mm(input ** 2, var_w.T) + self.bias.sigma ** 2
            eps = torch.randn(size=(var_b.size()), device=DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps

        else:
            z_k, _ = self.sample_z(input.size(0))
            e_w = self.weight.mu * self.alpha_q
            activations = torch.mm(input * z_k, e_w.T) + self.bias.mu

        if self.training or calculate_log_probs:

            z2, log_det_q = self.sample_z()
            W_mean = z2 * self.weight.mu * self.alpha_q
            W_var = self.weight.sigma **2 * self.alpha_q **2
            log_q0 = (-0.5 * torch.log(torch.tensor(math.pi)) - 0.5 * self.q0_log_var
                      - 0.5 * ((self.z - self.q0_mean) ** 2 / self.q0_log_var.exp())).sum()
            log_q = -log_det_q + log_q0

            act_mu = self.r0_c @ W_mean.T
            act_var = self.r0_c ** 2 @ W_var.T
            act_inner = act_mu + act_var.sqrt() * torch.randn_like(act_var)
            act = torch.tanh(act_inner)
            mean_r = self.r0_b1.outer(act).mean(-1)  # eq (9) from MNF paper
            log_var_r = self.r0_b2.outer(act).mean(-1)  # eq (10) from MNF paper
            z_b, log_det_r = self.r_flow(z2)
            log_rb = (-0.5 * torch.log(torch.tensor(math.pi)) - 0.5 * log_var_r
                      - 0.5 * ((z_b[-1] - mean_r) ** 2 / log_var_r.exp())).sum()
            log_r = log_det_r + log_rb


            kl_bias = (torch.log(self.bias_sigma_prior / self.bias.sigma) - 0.5 + (self.bias.sigma ** 2
                                + ( self.bias.mu - self.bias_mu_prior) ** 2) / (
                                   2 * self.bias_sigma_prior ** 2)).sum()

            kl_weight = (self.alpha_q * (torch.log(self.sigma_prior / self.weight.sigma)
                                         - 0.5 + torch.log(self.alpha_q / self.alpha_prior)
                                         + (self.weight.sigma ** 2 + (self.weight.mu * z2 - self.mu_prior) ** 2) / (
                                                     2 * self.sigma_prior ** 2))
                         + (1 - self.alpha_q) * torch.log((1 - self.alpha_q) / (1 - self.alpha_prior))).sum()

            self.kl = kl_bias + kl_weight + log_q - log_r

        else:
            self.kl = 0
        # propogate
        return activations

    # deine the whole BNN


class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.l1 = BayesianLinear(p - 1, 1, num_transforms=2)
        self.loss = nn.BCELoss(reduction='sum')

    def forward(self, x, sample=False):
        x = self.l1(x, sample)
        x = torch.sigmoid(x)
        return x

    def kl(self):
        return self.l1.kl


def train(net, optimizer, batch_size=BATCH_SIZE):
    net.train()
    old_batch = 0
    accs = []
    for batch in range(int(np.ceil(dtrain.shape[0] / batch_size))):
        batch = (batch + 1)
        _x = dtrain[old_batch: batch_size * batch, 0:p - 1]
        _y = dtrain[old_batch: batch_size * batch, -1]
        old_batch = batch_size * batch
        target = Variable(_y).to(DEVICE)
        data = Variable(_x).to(DEVICE)
        net.zero_grad()
        outputs = net(data, sample=True)
        target = target.unsqueeze(1).float()
        negative_log_likelihood = net.loss(outputs, target)
        loss = negative_log_likelihood + net.kl() / NUM_BATCHES
        loss.backward()
        optimizer.step()
        pred = outputs.squeeze().detach().cpu().numpy()
        pred = np.round(pred, 0)
        acc = np.mean(pred == _y.detach().cpu().numpy())
        accs.append(acc)

    print('loss', loss.item())
    print('nll', negative_log_likelihood.item())
    print('accuracy =', np.mean(accs))
    return negative_log_likelihood.item(), loss.item()





print("Classes loaded")
k = 100
predicted_alphas = np.zeros(shape=(k, 20))
true_alphas = np.array([0.97, 0.36, 0.40, 0.88, 0.46, 0.29, 1., 0.31, 0.61, 0.44,
                        0.91, 0.35, 1., 0.44, 0.35, 1., 1., 1., 1., 0.37])

true_weights2 = np.array([-4., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.2, 0., 37.1, 0., 0., 50., -0.00005, 10., 3., 0.])

true_weights = np.array([-4, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1.2, 0, 37.1, 0, 0, 50, - 0.00005, 10, 3, 0])

true_alphas = np.array([true_alphas, ] * k)
true_weights = np.array([true_weights, ] * k)

for i in range(0, k):
    print('model',i)
    torch.manual_seed(i)
    net = BayesianNetwork().to(DEVICE)
    optimizer = optim.Adam([
        {'params': net.l1.bias_mu, 'lr': 0.01},
        {'params': net.l1.bias_rho, 'lr': 0.01},
        {'params': net.l1.weight_mu, 'lr': 0.01},
        {'params': net.l1.weight_rho, 'lr': 0.01},
        {'params': net.l1.q0_mean, 'lr': 0.01},
        {'params': net.l1.q0_log_var, 'lr': 0.01},
        {'params': net.l1.r0_c, 'lr': 0.01},
        {'params': net.l1.r0_b1, 'lr': 0.01},
        {'params': net.l1.r0_b2, 'lr': 0.01},
        {'params': net.l1.lambdal, 'lr': 0.01},
        {'params': net.l1.z_flow.parameters(), 'lr': 0.004},
        {'params': net.l1.r_flow.parameters(), 'lr': 0.0035},

    ], lr=0.01)
    for epoch in range(epochs):
        if epoch == 50:
            optimizer = optim.Adam([
                {'params': net.l1.bias_mu, 'lr': 0.01},
                {'params': net.l1.bias_rho, 'lr': 0.01},
                {'params': net.l1.weight_mu, 'lr': 0.01},
                {'params': net.l1.weight_rho, 'lr': 0.01},
                {'params': net.l1.q0_mean, 'lr': 0.01},
                {'params': net.l1.q0_log_var, 'lr': 0.01},
                {'params': net.l1.r0_c, 'lr': 0.01},
                {'params': net.l1.r0_b1, 'lr': 0.01},
                {'params': net.l1.r0_b2, 'lr': 0.01},
                {'params': net.l1.lambdal, 'lr': 0.01},
                {'params': net.l1.z_flow.parameters(), 'lr': 0.003},
                {'params': net.l1.r_flow.parameters(), 'lr': 0.003},

            ], lr=0.01)
        if epoch == 100:
            optimizer = optim.Adam([
                {'params': net.l1.bias_mu, 'lr': 0.001},
                {'params': net.l1.bias_rho, 'lr': 0.001},
                {'params': net.l1.weight_mu, 'lr': 0.001},
                {'params': net.l1.weight_rho, 'lr': 0.001},
                {'params': net.l1.q0_mean, 'lr': 0.001},
                {'params': net.l1.q0_log_var, 'lr': 0.001},
                {'params': net.l1.r0_c, 'lr': 0.001},
                {'params': net.l1.r0_b1, 'lr': 0.001},
                {'params': net.l1.r0_b2, 'lr': 0.001},
                {'params': net.l1.lambdal, 'lr': 0.001},
                {'params': net.l1.z_flow.parameters(), 'lr': 0.002},
                {'params': net.l1.r_flow.parameters(), 'lr': 0.002},

            ], lr=0.01)
        if epoch == 450:
                optimizer = optim.Adam([
                    {'params': net.l1.bias_mu, 'lr': 0.001},
                    {'params': net.l1.bias_rho, 'lr': 0.001},
                    {'params': net.l1.weight_mu, 'lr': 0.001},
                    {'params': net.l1.weight_rho, 'lr': 0.001},
                    {'params': net.l1.q0_mean, 'lr': 0.001},
                    {'params': net.l1.q0_log_var, 'lr': 0.001},
                    {'params': net.l1.r0_c, 'lr': 0.001},
                    {'params': net.l1.r0_b1, 'lr': 0.001},
                    {'params': net.l1.r0_b2, 'lr': 0.001},
                    {'params': net.l1.lambdal, 'lr': 0.001},
                    {'params': net.l1.z_flow.parameters(), 'lr': 0.0004},
                    {'params': net.l1.r_flow.parameters(), 'lr': 0.0004},

                ], lr=0.01)
        print('epoch =', epoch)
        nll, loss = train(net, optimizer)
    net.l1.alpha = 1 / (1 + torch.exp(- net.l1.lambdal))
    predicted_alphas[i] = net.l1.alpha.data.detach().cpu().numpy().squeeze()
    a = net.l1.alpha.data.detach().cpu().numpy().squeeze()
    aa = np.round(a,0)
    tr = (true_weights2 != 0) * 1
    print(np.mean(aa == tr),'acc')
    # print(net.l1.weight.mu.data,'w')
    print(net.l1.alpha.data[0,0:6],'first 6 alphas')
    print(net.l1.alpha.data[0,10],'beta 11',net.l1.alpha.data[0,18],'beta19')

rmse2 = np.sqrt((np.square(predicted_alphas - true_alphas)).mean(axis=0)) * 100
print(np.mean(rmse2))

pa = np.round(predicted_alphas, 0)
tw = (true_weights != 0) * 1
print((pa == tw).mean(axis=1))
print((pa == tw).mean(axis=0))
np.savetxt('RMSE' +'.txt',rmse2, delimiter=',',fmt='%s')
np.savetxt('predicted_alphas' +'.txt',pa, delimiter=',',fmt='%s')
np.savetxt('true_weights' +'.txt',tw, delimiter=',',fmt='%s')



