import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./cifar10dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


writer = SummaryWriter("logs")

step = 0
tudui = Tudui()
# print(tudui)
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    # print(output.shape)
    print(imgs.shape)
    print(output.shape)
    # 张量的形状通常遵循 (N, C, H, W) 的格式，这里的 N 是批量大小（batch size），C 是通道数（number of channels），H 是图像的高度（height），W 是图像的宽度（width）
    # add_images记得加s
    writer.add_images("input", imgs, step)
    # torch.Size([64, 3, 32, 32])

    output = torch.reshape(output, (-1, 3, 30, 30))
    # add_images记得加s
    writer.add_images("output", output, step)
    # torch.Size([64, 6, 30, 30])
    step += 1

writer.close()
