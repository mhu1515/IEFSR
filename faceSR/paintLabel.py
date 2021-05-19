'''*-coding:utf-8 *-
 @Time     : 2020/11/2521:53
 @Author   : florrie(zfh)
'''
from PyQt5.QtGui import QImage, QPixmap
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys


class MyLabel(QLabel):
    def __init__(self, rect):
        super(MyLabel, self).__init__()
        self.rect = rect
        self.lab1 = QLabel(self)
        self.lab1.setStyleSheet("QLabel{color:rgb(0,0,255);font-size:18px;font-family:Arial;background:transparent;}")
        self.lab1.setMaximumSize(386, 386)

    # 绘制事件
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        painter.drawRect(self.rect)

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(675, 500)
        self.move(100, 50)
        self.setWindowTitle('在label中绘制矩形')
        self.lb = MyLabel(QRect(10,10,100,100))  # 重定义的label
        img = cv2.imread('./images/1.png')
        showImage = QImage(img.data, img.shape[1], img.shape[0],
                           QImage.Format_RGB888)  # 把读取到的图片数据变成QImage形式
        # 往显示视频的Label里 显示QImage
        self.lb.setPixmap(QPixmap.fromImage(showImage))
        # self.lb.setCursor(Qt.CrossCursor)
        self.horizontalLayout = QHBoxLayout(self)
        self.horizontalLayout.addWidget(self.lb)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    x = Example()
    sys.exit(app.exec_())