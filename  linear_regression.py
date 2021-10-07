import torch
import numpy as np

'''                      y = w1*x1+w2*x2+b          '''

# 生成数据
num_input = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 5
features = torch.tensor(np.random.normal(0,1,(num_examples,num_input)),dtype = torch.float)
labels = true_w[0]* features[:,0] +true_w[1] * features[:,1]+true_b
# 噪声
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float)

# 读取数据
import torch.utils.data as Data

# 批次为10，即1000个数据每批100个
batch_size = 10

dataset = Data.TensorDataset(features,labels)

data_iter = Data.DataLoader(dataset, batch_size,shuffle=True)
# for x,y in data_iter:
#     print(x,y)
#     break

# 定义模型
import torch.nn as nn

class LinearNet(nn.Module):
    def __init__(self,n_feature):
        super(LinearNet,self).__init__()

        #有多层时均可在此定义
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_input)
print(net) # 使⽤print可以打印出⽹络的结构

# 可以通过 net.parameters() 来查看模型所有的可学习参数，此函数将返回⼀个⽣成器。
for param in net.parameters():
    print(param)

# 初始化模型参数
from torch.nn import init

# init.normal_(net.weight, mean=0, std=0.01)
# init.constant_(net.bias, val=0)
for name,param in net.named_parameters():
    if 'weight' in name:
	    init.normal_(param,mean=0,std=0.1)

loss = nn.MSELoss()

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

for name,item in net.named_parameters():
    print("the name is ",name)
    print("the parameters is ",item)