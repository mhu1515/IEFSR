# USAGE
# python adjust_gamma.py

# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
import os

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# 读取文件夹中的所有图像，输入参数是文件名
def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        print(filename)  # 仅仅是为了测试
        original = cv2.imread(directory_name + "/" + filename)
        adjusted = adjust_gamma(original, gamma=0.2)
        # loop over various values of gamma
        # for gamma in np.arange(0.0, 3.5, 0.5):
        #     # ignore when gamma is 1 (there will be no change to the image)
        #     if gamma == 1:
        #         continue
        #     # apply gamma correction and show the images
        #     gamma = gamma if gamma > 0 else 0.1
        #     adjusted = adjust_gamma(original, gamma=gamma)
        #####显示图片#######
        # cv2.imshow(filename, adjusted)
        # cv2.waitKey(0)
        #####保存图片#########
        cv2.imwrite("./dataset/train/HR_dark/" + filename, adjusted)
        # return filename
read_directory("./dataset/train/HR/HR256/")
