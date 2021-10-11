import torch
import numpy as np
import d2lzh_pytorch as d2l

'''   y=1.2x-3.4x²+5.6x³+5+e   '''

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn(n_train + n_test, 1)  # torch.randn(size)用来生成随机数字的tensor，这些随机数字满足标准正态分布（0~1）。
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), dim=1)
# print(poly_features.shape) (200,3)
e = torch.tensor(np.random.normal(0, 0.01, size=n_train + n_test), dtype=torch.float)
labels = true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] + true_w[2] * poly_features[:,
                                                                                         2] + true_b + e
print(labels.size())

# 该方法已放入d2l
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None,
             y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    # d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    d2l.plt.show()


num_epochs = 100
loss = torch.nn.MSELoss()

print(poly_features[:n_train, :].shape[-1])


def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(3, 1)
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for epoch_i in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
        print(' epoch', epoch_i, ': train loss', train_ls[-1], 'test loss', test_ls[-1])
    print('weight:', net.weight.data, '\nbias:', net.bias.data)
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])


fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])
