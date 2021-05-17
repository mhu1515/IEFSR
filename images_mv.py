import shutil
import os
from glob import glob
'''
从"img_path"文件夹随机选取2000的图片移动到"copy_to_path"文件夹中
'''

# ROOT_DIR = os.path.abspath("../")
copy_to_path = "./dataset/FFHQ/val/HR/"
img_path = "./dataset/FFHQ/FFHQ-256/"
# img_path= "D:/code/celebA_hq/CelebA-HQ-img/"
# copy_to_path = "D:/code/celebA_hq/celeba-1024/"
data_names = os.listdir("./dataset/FFHQ/FFHQ-256/")

for i in range(100):
    img = data_names[i]
    # print(os.path.join(copy_to_path, '%05d' % int(img[:-4]) +'.jpg'))
    # shutil.copy(os.path.join(img_path, img), os.path.join(copy_to_path, '%05d' % int(img[:-4]) +'.jpg'))
    shutil.copy(os.path.join(img_path, img), os.path.join(copy_to_path, img))
    os.remove(os.path.join(img_path, img))  # remove
