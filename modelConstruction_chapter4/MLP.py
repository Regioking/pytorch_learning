import torch
from torch import nn
import numpy as np


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)
    def forward(self,X):
        hidden = self.hidden(X)
        a = self.act(hidden)
        return self.output(a)

X = torch.tensor(np.random.normal(0,0.01,(2,784)),dtype=torch.float)
net = MLP()
print(net)
print(net(X))
