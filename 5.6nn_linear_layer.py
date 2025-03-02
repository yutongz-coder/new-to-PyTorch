import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./cifar10dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


# 特征提取的含义：每个卷积核就像是一个特征探测器，它在输入图像上滑动进行卷积操作，以检测特定的特征。
# 例如，一个卷积核可能对图像中的边缘特征敏感，另一个卷积核可能对纹理特征敏感。通过使用 6 个不同的卷积核，卷积层就可以从输入图像中提取出 6 种不同类型的特征，每种特征对应一个特征图。
# 假设你有 64 张彩色的猫和狗的图像作为输入。经过卷积层处理后，对于每一张图像，都会得到 6 个特征图。
# 其中一个特征图可能突出显示了图像中动物的边缘，另一个特征图可能捕捉到了动物毛发的纹理，以此类推。这些特征图可以帮助后续的网络层进一步分析和识别图像中的内容。
# 综上所述，6 表示的是卷积层为每张输入图像提取的不同特征的数量，虽然可以类比为通道，但它代表的是从图像中提取的特征信息，而不是像 RGB 那样的颜色信息。
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)
# out_features 的值为 10，这意味着经过 self.linear1 层的线性变换后，每个样本会输出 10 个特征

    def forward(self, input):
        output = self.linear1(input)
        return output


tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # torch.flatten(imgs) 函数将 imgs 张量展平成一个一维向量。由于 imgs 的形状是 (64, 3, 32, 32)，展平后，元素的总数为 64 * 3 * 32 * 32 = 196608
    # 所以 output = torch.flatten(imgs) 的形状为 (196608,)
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)

