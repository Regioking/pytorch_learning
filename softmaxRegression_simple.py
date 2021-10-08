import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import d2lzh_pytorch as d2l
import numpy as np

import torch.nn as nn
from torch.nn import init

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                                train=True, download=True, transform=transforms.ToTensor())

''' transforms.ToTensor() 将尺⼨为 (H x W x C) 且数据位于[0, 255]的 PIL 图⽚或者数据类型为 np.uint8 的 NumPy 数组转换
 为尺⼨为 (C x H x W) 且数据类型为 torch.float32 且位于[0.0, 1.0]的 Tensor'''
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                               train=False, download=True, transform=transforms.ToTensor())
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])
# d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不⽤额外的进程来加速读取数据,其他的貌似会出问题
else:
    num_workers = 4

train_iter = torch.utils.data.DataLoader(mnist_train,
                                         batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,
                                        batch_size=batch_size, shuffle=False, num_workers=num_workers)

num_inputs = 784
num_outputs = 10


# W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype = torch.float, requires_grad = True)
# b = torch.zeros(num_outputs, dtype=torch.float,)
# b.requires_grad=True
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):  # x shape: (batch,1,28,28)
        y = self.linear(x.view(x.shape[0], -1))  # -1会自动计算
        return y


net = LinearNet(num_inputs, num_outputs)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 5


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零 ， 清的是 w,b的梯度
            optimizer.zero_grad()
            # 算梯度
            l.backward()
            # 更新 w,b
            optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


num_epochs = 5
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])
