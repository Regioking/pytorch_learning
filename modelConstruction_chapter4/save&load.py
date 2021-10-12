import torch
from torch import nn

''' Tensor save&load  '''
x = torch.ones(3)
torch.save(x, 'x.pt')

x2 = torch.load('x.pt')
print(x2)

y = torch.zeros(4)
torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
print(xy)

'''Model save&load 
1. 仅保存和加载模型参数( state_dict )(推荐⽅式)；state_dict 是⼀个从参数名称映射到参数 Tesnor 的字典对象。
torch.save(model.state_dict(), PATH)

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))

2. 保存和加载整个模型。
torch.save(model, PATH)
model = torch.load(PATH)
'''


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


net = MLP()
# 只有具有可学习参数的层(卷积层、线性层等)才有 state_dict 中的条⽬。优化器( optim )也有
# ⼀个 state_dict ，其中包含关于优化器状态以及所使⽤的超参数的信息。
print(net.state_dict())
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())

X = torch.randn(2, 3)
Y = net(X)
PATH = "./net.pt"
torch.save(net.state_dict(), PATH)
net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
# 因为这 net 和 net2 都有同样的模型参数，那么对同⼀个输⼊ X 的计算结果将会是⼀样的。
print((Y2 == Y))

