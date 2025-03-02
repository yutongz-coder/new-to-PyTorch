import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
# print("ok")
# print(vgg16_true)

# 用现有的网络改动vgg的结构，即把vgg当成前置的网络
train_data = torchvision.datasets.CIFAR10('./cifar10dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
# 在vgg16模型的基础上再加一层
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# 修改原有模型，原有模型结构：
# print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)




