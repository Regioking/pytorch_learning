import torch
import numpy as np
import d2lzh_pytorch as d2l

# Fashion-MNIST
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256
'''
 784  *  256  *  10
'''
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
params = [W1, b1, W2, b2]


def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))


def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.mm(X, W1) + b1)
    return torch.mm(H, W2) + b2


loss = torch.nn.CrossEntropyLoss()
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
