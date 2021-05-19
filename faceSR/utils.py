'''*-coding:utf-8 *-
 @Time     : 2020/11/269:18
 @Author   : florrie(zfh)
'''
import numpy as np
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

import io
import cv2
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QBuffer
from PIL import Image, ImageQt
import math

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def qimage_to_cvimg(qimage):
    '''
    @description: 将QImage图片转换为CV2 Mat类型
    @param {QImage}: 需要转换的图片
    @return: 转换后的CV2 Mat类型图片
    '''

    qimage = qimage.convertToFormat(4)
    ptr = qimage.constBits()
    ptr.setsize(qimage.byteCount())
    cv_img = np.array(ptr).reshape(qimage.height(), qimage.width(), 4)

    return cv_img


def cvimg_to_qimage(cv_img):
    '''
    @description: 将CV2 Mat图片转换为QImage
    @param {Mat}
    @return: 转换后的QImage图片
    '''
    h, w, d = cv_img.shape
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    qimage = QImage(cv_img.data, w, h, w * 3, QImage.Format_RGB888)

    return qimage


def qimage_to_pilimg(img):
    '''
    description: 将QImage转换为PIL Image
    param {QImage}
    return {PIL Image}
    '''
    logger.debug("将QImage转换为PIL Image")
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    img.save(buffer, "PNG")
    pil_im = Image.open(io.BytesIO(buffer.data()))
    return pil_im



def pilimg_to_qimage(pilimg):
    '''
     description: 将PIL Image转换为QImage
     param {PIL Image}
     return {QImage}
     '''
    return ImageQt.ImageQt(pilimg)


def qimage_to_ndarray(img):
    # pil_img = qimage_to_pilimg(img)
    pil_img = ImageQt.fromqimage(img)

    img_arr = np.array(pil_img)
    '''
    if img_arr.ndim == 2:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
    else:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    '''
    return img_arr

# ----------------------- 常用图像质量评价指标-------------------------------

def compute_psnr(im1, im2):
    p = psnr(im1, im2)
    return p


def compute_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s


def entropy_My(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype='float64')
    rows, cols = gray.shape
    # hist代表0-255个灰度级的对应的频数
    bin = 256
    hist, bins = np.histogram(gray, bin)
    res = 0
    for i in range(bin):
        p = hist[i]/(rows*cols)
        if(p!=0):
            res -= p*(math.log(p) / math.log(2.0))
    print('信息熵：' ,res)
    return res

def avgLightImg(image):
    '''
    亮度：即图像矩阵的平均值，其值表明图像的明暗程度
    :param image: 彩色图
    :return:
    '''
    image = np.array(image, dtype='float64')
    avgb = image[:, :, 0].mean()
    avgg = image[:, :, 1].mean()
    avgr = image[:, :, 2].mean()
    L = (1/3)*(avgb+avgg+avgr)
    print("图像亮度（均值）", L)
    return L

def stdImg(image):
    '''
    标准差：反映图像中黑白反差的程度，越大，说明对比度越大
    :param image: 灰度图
    :return:
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = image.std()
    print("图像标准差", std)
    return std

def contrastAddImg(origin, image, size=3):
    '''
    对比度增量：度量增强后图像的对比度与原图对比度关系，反映图像变化
    前后对比度的变化程度，如果大于1，则说明有增强
    :param origin:
    :param image:
    :return:
    '''
    X = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    Y = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    w,h = X.shape
    #if (w%size!=0):

def avgGradient(img):
    '''
    计算图像的平均梯度,平均梯度反映图像清晰度和纹理变化，
    梯度越大，说明图像越清晰
    :param img:
    :return:
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype='float64')
    tmp = 0
    rows, cols = gray.shape
    for i in range(rows-1):
        for j in range(cols-1):
            dx = gray[i, j+1] - gray[i,j]
            dy = gray[i+1, j] - gray[i, j]
            ds = np.sqrt((dx**2+dy**2)/2)
            tmp += ds

    gradient = tmp / (rows*cols)
    print('图像梯度为：', gradient)
    return gradient
