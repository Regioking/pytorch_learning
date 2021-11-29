import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import matplotlib.pyplot as plt
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = 'data'
# ImageFolder : 相当于一个DataLoader
train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))

# data: https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/hotdog.zip
'''train/
     |---hotdog
     |---not-hotdog
'''


def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()


hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
# show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# data1 = transforms.RandomResizedCrop(224, scale=(1.2, 1.4))(
#     train_imgs[0][0])  # scale 感觉跟截取图片的位置有关，比如原图 w:h=1.2:1.4,设置相同scale后 截出的形状基本不变，或者理解为在高宽上截取的比例
# data2 = transforms.CenterCrop(224)(train_imgs[0][0])  # 从中间截，固定的
# data3 = transforms.CenterCrop(224)(train_imgs[0][0])
# plt.subplot(2, 2, 1), plt.imshow(train_imgs[0][0]), plt.title("原图")
# plt.subplot(2, 2, 2), plt.imshow(data1), plt.title("转换后的图1")
# plt.subplot(2, 2, 3), plt.imshow(data2), plt.title("转换后的图2")
# plt.subplot(2, 2, 4), plt.imshow(data3), plt.title("转换后的图3")
# plt.show()

train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])

test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize])
pretrained_net = models.resnet18(pretrained=True)
nn.Linear(in_features=512, out_features=1000, bias=True)
print(pretrained_net.fc)
# 将最后的fc 成修改我们需要的输出类别数.如果你使⽤的是其他模型，那可能没有成员变量fc（⽐如models中的VGG预训练模型），所以正确做法是查看对应模型源码中其定义部分
pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)

output_params = list(map(id, pretrained_net.fc.parameters()))

feature_params = filter(lambda p: id(p) not in output_params,
                        pretrained_net.parameters())
lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}], lr=lr, weight_decay=0.001)

import d2lzh_pytorch as d2l


def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs), batch_size,
                            shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs), batch_size)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train_ch5(net, train_iter, test_iter, optimizer, device, num_epochs, loss)


train_fine_tuning(pretrained_net, optimizer)
