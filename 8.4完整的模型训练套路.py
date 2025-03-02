# 利用已经训练好的模型进行测试
# 如果是png格式的图片的话，还需要加一个image = image.convert('RGB')，因为png格式是四个通道，除了RGB三通道外，还有一个透明度通道
# 如果图片本来就是三个颜色通道，经过此操作，不变
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = r"D:\python\learnPytorch\images\57afd199bdb19a9345c3f6916bb8de6.jpg"
image = Image.open(image_path)
image = image.convert('RGB')
print(image)
# image.show()

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)


# 加载网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5,1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 如果是用Google Colab跑的程序，需要下载相应的pth文件到此文件夹中，然后输入：
model = torch.load("tudui_29.pth", map_location=torch.device('cpu'))

# 用的是8.1train_model里面的训练了几轮保存的模型
# model = torch.load("tudui_29.pth")
print(model)

image = torch.reshape(image, (1, 3, 32, 32))
# 模型测试
model.eval()
# 节约内存
with torch.no_grad():
    output = model(image)
print(output)
# 找出每个样本得分最高的类别索引
# im = 1 表示在类别维度上进行查找，即针对每个样本找出得分最高的类别。_ 用于接收最大值，而 predicted 接收最大值对应的索引，也就是预测的类别索引
_, predicted = torch.max(output, 1)
# 输出预测类别
print("预测的类别索引:", predicted)


