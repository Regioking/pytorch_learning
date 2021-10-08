import torch
import torch.nn as nn
from torch.nn import init
import d2lzh_pytorch as d2l


class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


#num_inputs, num_outputs, num_hiddens = 784, 10, 256


class MLPNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super(MLPNet, self).__init__()
        self.model = nn.Sequential(
            FlattenLayer(),
            nn.Linear(num_inputs, num_hiddens),
            nn.LeakyReLU(),
            nn.Linear(num_hiddens, num_outputs),
        )

    def forward(self, X):
        X = self.model(X)
        return X


net = MLPNet(784, 10, 256)

for name, params in net.model.named_parameters():
    if "weight" in name:
        init.normal_(params, mean=0, std=0.01)
    if "bias" in name:
        init.constant_(params, val=0)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
