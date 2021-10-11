from torch import nn
import torch
import numpy as np
from collections import OrderedDict


class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
print(net)
X = torch.tensor(np.random.normal(0, 0.01, (2, 784)), dtype=torch.float)
print(net(X))

'''ModuleDict 接收⼀个⼦模块的字典作为输⼊, 然后也可以类似字典那样进⾏添加访问操作:'''

net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net.output = nn.Linear(256, 10)  # 添加
# net['output'] = nn.Linear(256, 10) # 另一种添加方式
print(net['linear'])  # 访问
print(net.output)
print(net)

net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10))  # # 类似List的append操作
print(net[-1])  # 类似List的索引访问
print(net)
