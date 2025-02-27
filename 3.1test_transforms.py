import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "dataset/train/bees_image/198508668_97d818b6c4.jpg"
img = Image.open(img_path)

# 检查图像类型
print("图像类型:", type(img))

writer = SummaryWriter("logs")

# 1.transforms该如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# 检查转换后的数据类型
print("转换后的数据类型:", type(tensor_img))
print(tensor_img)
print(tensor_img.shape)

writer.add_image("tensor_img", tensor_img)

writer.close()
