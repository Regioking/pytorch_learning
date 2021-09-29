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

for x,y in data_iter:
    print(x,y)
    break


