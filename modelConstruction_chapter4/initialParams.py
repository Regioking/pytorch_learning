import torch
from torch import nn
'''自定义初始化函数'''


def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()


net = torch.nn.Linear(4, 3);
net.add_module("output", torch.nn.Linear(3, 1))

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)

'''共享模型参数'''

linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    nn.init.constant_(param, val=3)
    print(name, param.data)
