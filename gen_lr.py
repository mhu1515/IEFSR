import os
from glob import glob
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms

from PIL import Image, ImageOps
import torchvision.transforms.transforms as transforms


def read_image(file_name, image_height=32, image_width=32):
    image = Image.open(file_name)
    image = np.asarray(image)
    image = Image.fromarray(image)
    image = transforms.Resize((image_width, image_height))(image)
    image_array = np.array(image)
    image_output = Image.fromarray(image_array)
    # image = np.asarray(image)
    # image = image.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    image_output.save(os.path.join('./test/', 'lrpil_' +file_name.split('/')[-1][-10:]))
    # print(image)
    return image
# 无损保存jpg图像
# 从中间框256*256
def gen_lr():
    all_list = glob('E:/YouTube/FaceForensics/test_2/*.jpg')
    output_path = 'E:/YouTube/FaceForensics/test_2_32/'

    for i in all_list:

        input = Image.open(i)
        width, height = input.size
        if width == height:
            region = input
        else:
            if width > height:
                delta = (width - height) / 2
                box = (delta, 0, delta + height, height)
            else:
                delta = (height - width) / 2
                box = (0, delta, width, delta + width)
            region = input.crop(box)
        # image_resize = region.resize((256, 256),Image.ANTIALIAS)
        image_resize = input.resize((32, 32), Image.BILINEAR)
        image_array = np.array(image_resize)
        # array to image
        image_output = Image.fromarray(image_array)
        # print(os.path.join(output_path,i.split('/')[-1].split('\\')[-1]))
        # image_output.save(os.path.join(output_path, i.split('/')[-1][-10:]), 'PNG', quality=100)
        image_output.save(os.path.join(output_path,i.split('/')[-1].split('\\')[-1]),'PNG',quality=100)

def saving():
    all_list = glob('./dataset/celeba-256/train/*')
    output_path = './dataset/celeba-256/train_bililiar_lr/'

    for i in all_list:

        input = Image.open(i)
        image_resize = input.resize((256, 256), Image.ANTIALIAS)
        # image_resize = input.resize((32, 32),Image.BILINEAR)
        image_array = np.array(image_resize)
        # array to image
        image_output = Image.fromarray(image_array)
        image_output.save(os.path.join(output_path,i.split('/')[-1][-10:]))

def singleImage(path,output_path):
    input = Image.open(path)
    # print(np.array(input))
    # image_resize = input.resize((256, 256), Image.ANTIALIAS)
    image_resize = input.resize((32, 32), Image.BILINEAR)
    image_array = np.array(image_resize)
    # print(np.array(image_resize))
    # array to image
    image_output = Image.fromarray(image_array)
    # print(path.split('/')[-1].split('\\')[-1])
    image_output.save(os.path.join(output_path,path.split('/')[-1].split('\\')[-1]),'PNG',quality=100)

# all_list = glob('E:/YouTube/FaceForensics/test_2/*.jpg')
# output_path = 'E:/YouTube/FaceForensics/test_2_32/'
all_list = glob('./dataset/train/HR_enhanced/*.jpg')
output_path = './dataset/train/enlighten/LR_enhanced/'
for path in all_list:
    singleImage(path,output_path)
# gen_lr()
