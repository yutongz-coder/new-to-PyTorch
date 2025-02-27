import torch
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# 打开一张图像
image_path = "dataset/train/ants_image/6743948_2b8c096dda.jpg"
img = Image.open(image_path)

# 创建调整大小变换对象
trans_resize = transforms.Resize(512)
# 创建随机裁剪变换对象
trans_random = transforms.RandomCrop(512)
# 创建转换为张量的变换对象
trans_totensor = transforms.ToTensor()
# 组合变换
trans_compose_2 = transforms.Compose([trans_resize, trans_random, trans_totensor])

# 创建 SummaryWriter 对象
writer = SummaryWriter()

# 循环处理图像并记录到 TensorBoard
for i in range(10):
    img_crop = trans_compose_2(img)
    print(f"图像类型: {type(img_crop)}, 图像形状: {img_crop.shape}, 图像最小值: {img_crop.min()}, 图像最大值: {img_crop.max()}")
    writer.add_image("RandomCrop", img_crop, i)

# 关闭 SummaryWriter
writer.close()
