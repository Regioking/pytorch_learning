import torch
import torch.nn as nn
# from torch import nn
import numpy as np


class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)  # 常参
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.mm(x, self.rand_weight) + 1)
        # 复⽤全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这⾥我们需要调⽤item函数来返回标量进⾏⽐较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


'''因为 FancyMLP 和 Sequential 类都是 Module 类的⼦类，所以我们可以嵌套调⽤它们。'''


class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)

net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())
X = torch.rand(2, 40)
print(net)
print(net(X))
