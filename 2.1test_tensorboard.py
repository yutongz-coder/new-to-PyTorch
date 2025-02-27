from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "dataset/train/ants_image/69639610_95e0de17aa.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("train", img_array, 1, dataformats='HWC')
print(type(img_array))
print(img_array.shape)
print(img_array)

for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)

writer.close()
