import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PropagateFlow(nn.Module):
    def __init__(self,transform,dim,num_transforms):
        super().__init__() 
        if transform == 'RNVP':
            self.transforms = nn.ModuleList([RNVP(dim) for i in range(num_transforms)])
        else:
            print('Transform not implemented')

    def forward(self, z):
        logdet = 0
        for f in self.transforms:
            z = f(z)
            logdet += f.log_det()
        return z, logdet


class MLP(nn.Sequential):
    """Multilayer perceptron"""

    def __init__(self, *layer_sizes, leaky_a=0.1):
        layers = []
        for s1, s2 in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Linear(s1, s2))
            layers.append(nn.LeakyReLU(leaky_a))
        super().__init__(*layers[:-1])  # drop last ReLU


class RNVP(nn.Module):

    def __init__(self, dim, h_sizes=[50,50,50,50,50]):
        super().__init__()
        self.network = MLP(*[dim] + h_sizes)
        self.t = nn.Linear(h_sizes[-1], dim)
        self.s = nn.Linear(h_sizes[-1], dim)
        self.eps = 1e-6
        self.gate = 0

    def forward(self, z):  # z -> x
        self.mask = torch.bernoulli(0.5 * torch.ones_like(z))
        z1, z2 = (1 - self.mask) * z, self.mask * z
        y = self.network(z2)
        shift, scale = self.t(y), self.s(y)
        self.gate = torch.sigmoid(scale)
        x = (z1 * self.gate + (1 - self.gate) * shift) + z2
        return x

    def log_det(self):
        return ((1 - self.mask) * self.gate.log()).sum(-1)






