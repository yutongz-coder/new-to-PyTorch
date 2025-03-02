import torch
import torchvision
from torch import nn
# 需要先把7.2文件的名字改成“model save"
from model_save import *

# 方式1-》保存方式1.加载模型，保存的时候模型的参数也被保存下来了
model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2.加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(model)


# 陷阱
model = torch.load("tudui_method1_pth")
print(model)
# 会报错
