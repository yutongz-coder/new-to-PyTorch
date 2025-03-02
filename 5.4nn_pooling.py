# stride的默认值是kernel的大小
# ceil_mode默认是false

import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./cifar10dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, 64)

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# input = torch.reshape(input, (-1, 1, 5, 5))
# # -1表示自动计算batch_size
# print(input.shape)

# 为什么要进行最大池化：将数据量减小，训练参数减少，训练速度更快
#  “2d” 代表二维（two - dimensional），表明该操作是在二维数据上进行的
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


tudui = Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter("logs_maxpool")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()

