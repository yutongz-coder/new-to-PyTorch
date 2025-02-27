from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img_path = "dataset/train/ants_image/28847243_e79fe052cd.jpg"
img = Image.open(img_path)
print(img)

# 1.ToTensor的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# 2.normalize
# input[channel]=(input[channel]-mean[channel])/std[channel]
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

# 3.resize
print(img.size)
trans_resize = transforms.Resize((300, 300))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(type(img_resize))
print(img_resize)

# 4.compose
trans_resize_2 = transforms.Resize(300)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# 4.RandomCrop
# 这里指定裁剪后的图像为正方形，边长为 512 像素。
trans_random = transforms.RandomCrop((200, 300))
# 当将一个图像应用 trans_compose_2 时，图像会按照列表中变换的顺序依次进行处理。首先，图像会被随机裁剪为 512×512 像素的区域，然后将裁剪后的图像转换为 torch.Tensor 类型。
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)


writer.close()



