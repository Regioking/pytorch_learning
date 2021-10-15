import torch
from torch import nn, optim
import time
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels,out_channels,kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernal_size ,stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        # print(feature.shape) (256,16,4,4)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


net = LeNet()
print(net)
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

lr, num_epochs = 0.001, 1
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

net = net.to(device)
print("training on", device)
loss = torch.nn.CrossEntropyLoss()
batch_count = 0
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
    for x, y in train_iter:
        #print(y.shape)  # (256)
        #print(x.shape)  # (256,1,28,28)
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        #print(y_hat.shape)  # (256,10)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.cpu().item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
        batch_count += 1
    test_acc = d2l.evaluate_accuracy(test_iter, net)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,time % .1fsec' % (
        epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
