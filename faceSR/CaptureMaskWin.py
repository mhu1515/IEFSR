'''
@Author: Me
@Date: 2020-07-30 23:46:57
LastEditTime: 2020-08-21 14:08:04
LastEditors: Please set LastEditors
@Description: 对视频帧进行截图的页面, 提供一个遮罩效果, 获取截取的图片
@FilePath: \SuperResolution\main\CaptureMaskWin.py
'''


import Debuger

import sys
from PyQt5.QtGui import QPainter, QIcon, QPixmap

from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QFileDialog, QSystemTrayIcon, QAction, QMenu,QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QRect
import logging

from PIL import Image, ImageQt
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class ScreenShotsWin(QWidget):
    # 定义一个信号
    oksignal = pyqtSignal()
    finsignal = pyqtSignal(QPixmap)
    # rectsignal = pyqtSignal(QRect)

    def __init__(self):
        super(ScreenShotsWin, self).__init__()
        self.initUI()
        self.start = (0, 0)  # 开始坐标点
        self.end = (0, 0)  # 结束坐标点

    def initUI(self):
        # self.showFullScreen()
        self.setWindowOpacity(0.4)  # 改变窗体的透明度
        self.oksignal.connect(lambda: self.screenshots(self.start, self.end))

    def screenshots(self, start, end):
        '''
        截图功能
        :param start:截图开始点
        :param end:截图结束点
        :return:
        '''
        logger.debug('开始截图,%s, %s', start, end)

        x = min(start[0], end[0])
        y = min(start[1], end[1])
        width = abs(end[0] - start[0])
        height = abs(end[1] - start[1])

        des = QApplication.desktop()
        screen = QApplication.primaryScreen()
        if screen:
            self.setWindowOpacity(0.0)
            pix = screen.grabWindow(des.winId(), x, y, width, height)

        print(type(pix))
        self.finsignal.emit(pix)
        # self.rectsignal.emit(QRect(x, y, width, height))
        self.close()

    def paintEvent(self, event):
        '''
        给出截图的辅助线
        :param event:
        :return:
        '''
        logger.debug('开始画图')
        x = self.start[0]
        y = self.start[1]
        w = self.end[0] - x
        h = self.end[1] - y

        pp = QPainter(self)
        pp.drawRect(x, y, w, h)

    def mousePressEvent(self, event):

        # 点击左键开始选取截图区域
        if event.button() == Qt.LeftButton:
            self.start = (event.pos().x(), event.pos().y())
            logger.debug('开始坐标：%s', self.start)

    def mouseReleaseEvent(self, event):

        # 鼠标左键释放开始截图操作
        if event.button() == Qt.LeftButton:
            self.end = (event.pos().x(), event.pos().y())
            logger.debug('结束坐标：%s', self.end)

            self.oksignal.emit()
            logger.debug('信号提交')
            # 进行重新绘制
            self.update()

    def mouseMoveEvent(self, event):

        # 鼠标左键按下的同时移动鼠标绘制截图辅助线
        if event.buttons() and Qt.LeftButton:
            self.end = (event.pos().x(), event.pos().y())
            # 进行重新绘制
            self.update()


class MyLabel(QLabel):
    def __init__(self, rect):
        super(MyLabel, self).__init__()
        self.rect = rect
        self.lab1 = QLabel(self)
        self.lab1.setStyleSheet("QLabel{color:rgb(0,0,255);font-size:18px;font-family:Arial;background:transparent;}")
        self.lab1.setMaximumSize(256, 256)

    # 绘制事件
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        painter.drawRect(self.rect)