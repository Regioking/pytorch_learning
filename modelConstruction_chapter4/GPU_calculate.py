import torch
from torch import nn

# GPU数量
print(torch.cuda.device_count())

if (torch.cuda.is_available()):
    # 查看当前GPU索引号，从0开始
    print(torch.cuda.current_device())
    # 根据索引号查看GPU名字
    print(torch.cuda.get_device_name(0))

    x = torch.tensor([1, 2, 3])
    # .cuda() 可以将CPU上的 Tensor 转换（复制）到GPU上
    # ⽤ .cuda(i)来表示第i块GPU及相应的显存  cuda(0) 和 cuda() 等价
    x = x.cuda(0)
    # 我们可以通过 Tensor 的 device 属性来查看该 Tensor 所在的设备。
    print(x.device)

device = torch.device('cuda' if torch.cuda.is_available() else
                      'cpu')
x = torch.tensor([1, 2, 3], device=device)
# or
x = torch.tensor([1, 2, 3]).to(device)
# 如果对在GPU上的数据进⾏运算，那么结果还是存放在GPU上。
y = x ** 2
print(y)

'''模型的GPU计算'''
net = nn.Linear(3, 1)
# 查看device
print(list(net.parameters())[0].device)

if (torch.cuda.is_available()):
    net.cuda()
    print(list(net.parameters())[0].device)
    x = torch.rand(2, 3).cuda()
    net(x)
