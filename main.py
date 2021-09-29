# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples)) #  feature shape(1000,2)   列出0-1000
    random.shuffle(indices) # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size,num_examples)]) # 最后⼀次可能不⾜⼀个batch
        # index_select(dim, list) , 在指定dim上选出索引为list的向量
        yield features.index_select(0, j), labels.index_select(0,
j)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
