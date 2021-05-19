'''*-coding:utf-8 *-
 @Time     : 2020/11/1118:25
 @Author   : florrie(zfh)
'''
import os
import sys
from PIL import Image, ImageQt
import numpy as np
from CaptureMaskWin import ScreenShotsWin, MyLabel
import time
from UIfile.GUIForm import Ui_MainWindow, MyFigure
from matplotlib.ticker import FuncFormatter


from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, \
    QLabel, QListWidgetItem, QMessageBox, QAction
from PyQt5.QtGui import QIcon, QImage, QPixmap, QPen, QPainter, QColor, QGuiApplication
from PyQt5.QtCore import QSize, QTimer, Qt, QUrl
from algoThread import algoThread

import Debuger
import utils

curr_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curr_path)[0]
sys.path.append(root_path)
print(root_path)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class FaceSRMainWindow(Ui_MainWindow, QMainWindow):

    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.img = None
        self.setWindowIcon(QIcon('UIfile/logo.png'))
        self.initUI(self)
        self.algorithm = "SFSR"
        self.scale = 4
        self.screenshot_act.triggered.connect(self.self_detect_clicked)

        # 点击选择文件 信号发射
        self.open_file.triggered.connect(self.btn_file_select)
        self.fileCho.clicked.connect(self.btn_file_select)
        # 当下来索引发生改变时发射信号触发绑定的事件
        self.algoCho.currentIndexChanged.connect(self.algo_select)
        # 填写scale值
        self.scaleCho.textChanged.connect(self.scale_edit)
        # ok按键点击，运行算法
        self.ok.clicked.connect(self.okfun)
        # cancel按键点击，重置状态
        self.reset.clicked.connect(self.resetting)
        # 定义线程
        self.algothread = algoThread()
        # 线程完成后发送的信号，连接到槽函数
        self.algothread.trigger.connect(self.finish_alogrithm)
        self.show()

    def btn_file_select(self):
        self.imgName, imgType = QFileDialog.getOpenFileName(self, '打开文件', os.getcwd(), "选择图像(*.jpg;*.png)")
        self.fileCho.setText(os.path.basename(self.imgName))
        print("你选择的文件：", self.imgName)
        # 设置文件
        self.img = Image.open(self.imgName)
        self.inp_img.setPixmap(self.img.toqpixmap())
        self.inp_img.setScaledContents(True)

    # 下拉选择算法
    def algo_select(self):
        self.algorithm = self.algoCho.currentText()
        print("select algorithm:", self.algorithm)

    # 填写scale
    def scale_edit(self, val):
        if (not val == '') and (val.isdigit()):
            self.scale = int(val)
            print("scale=", val, "type:", type(self.scale))

    # cancel键被点击，重置参数
    def resetting(self):
        print("重置参数....")
        self.scale = 4
        self.algorithm = "SFSR"
        self.fileCho.setText("点击选择图像上传")
        self.algoCho.setCurrentIndex(0)
        self.scaleCho.setText("4")
        self.img = None
        self.cat_img.setPixmap(QPixmap(""))
        self.out_img.setPixmap(QPixmap(""))

        self.table.setItem(0, 1, QTableWidgetItem("0.0"))
        self.table.setItem(1, 1, QTableWidgetItem("0.0"))
        self.table.setItem(2, 1, QTableWidgetItem("0.0"))
        self.table.setItem(3, 1, QTableWidgetItem("0.0"))
        self.table.setItem(4, 1, QTableWidgetItem("0.0"))
        self.table.setItem(5, 1, QTableWidgetItem("0.0"))
        if self.curvely.count() >1 :
            self.curvely.itemAt(0).widget().deleteLater()

    def self_detect_clicked(self):
        if self.img is None:
            reply = QMessageBox.information(self, "提示", "你没有上传图像！", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        else:
            self.screenshot = ScreenShotsWin()
            self.screenshot.showFullScreen()
            self.screenshot.finsignal.connect(self.finished)
            # self.screenshot.rectsignal.connect(self.paintRect)

    def finished(self, img):
        '''
        截图信号完成，将截到的小图设置为self.inp_img
        :param img:
        :return:
        '''
        logger.debug("截图信号发生")
        self.cat_img.setScaledContents(True)
        self.cat_img.setPixmap(img)

        a = img.toImage()
        self.catImg = utils.qimage_to_pilimg(a)

    def drawRect(self, rect):
        '''
        画矩形框（暂时不用）
        :param rect:
        :return:
        '''
        print("画图信号返回")
        self.inp_img = MyLabel(rect)  # 重定义的label
        # 往显示视频的Label里 显示QImage
        self.inp_img.setPixmap(self.img.toqpixmap())
        self.inp_img.setScaledContents(True)
        # self.inp_img.repaint()

    # ------------------------算法----------------------------------
    # ok键被点击，运行算法
    def okfun(self):
        if self.img is None:
            reply = QMessageBox.information(self, "提示", "你没有上传图像！", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        else:

            print("正在运行算法...")
            # 设置文件上传按钮不能按
            self.fileCho.setEnabled(False)
            # 设置scale框不能编辑（只读）
            self.scaleCho.setReadOnly(True)
            # 设置取消按钮不能按
            self.reset.setEnabled(False)
            if self.catImg is None:
                self.catImg = self.img

            if self.algorithm == "SFSR":
                self.modeName = "SFSR"
                print("mode :", self.algorithm)
            # elif self.algorithm == "VESPCN":
            #     self.modeName = "Mode2"
            #     print("mode 2:", self.algorithm)
            # elif self.algorithm == "SRCNN":
            #     self.modeName = "Mode4"
            # elif self.algorithm == "Bicubic":
            #     self.modeName = "Mode3"
            else:
                self.modeName = "SFSR"
                self.train_mode = 7
            self.start_time = time.time()  # 开始计时
            # 设置线程的scale参数
            self.algothread.setScale(self.scale)
            self.algothread.setInput(self.catImg)
            # 启动线程
            self.algothread.start()

    def finish_alogrithm(self, img):
        Debuger.debug("算法完成")
        self.out_img.setPixmap(img)
        self.out_img.setScaledContents(True)
        self.end_time = time.time()
        # 实时刷新界面
        QApplication.processEvents()

        self.aveLightVal.setText(str('%.6f'%(self.algothread.light)))
        self.aveStdVal.setText(str('%.6f'%(self.algothread.std)))
        self.aveInfVal.setText(str('%.6f'%(self.algothread.entropy)))
        self.aveGradVal.setText(str('%.6f'%(self.algothread.gradient)))
        self.aveTimeVal.setText(str('%.6f'%(self.end_time-self.start_time))+'s')
        self.remarkVal.setText(self.algorithm + "已完成")

        self.table.setItem(0, 1, self.aveLightVal)
        self.table.setItem(1, 1, self.aveStdVal)
        self.table.setItem(2, 1, self.aveInfVal)
        self.table.setItem(3, 1, self.aveGradVal)
        self.table.setItem(4, 1, self.aveTimeVal)
        self.table.setItem(5, 1, self.remarkVal)

        F = MyFigure(3, 3, 100)
        axes = F.fig.add_subplot(111)
        src = img.toImage()
        src = utils.qimage_to_pilimg(src)

        arr = np.array(src).flatten()
        axes.hist(arr, bins=256, density=1, facecolor='r', edgecolor='b')
        formatter = FuncFormatter(self.to_six)
        F.fig.gca().yaxis.set_major_formatter(formatter)
        F.fig.suptitle("Histogram")
        self.curvely.itemAt(0).widget().deleteLater()
        self.curvely.addWidget(F)

        # 设置文件上传按钮不能按
        self.fileCho.setEnabled(True)
        # 设置scale框不能编辑（只读）
        self.scaleCho.setReadOnly(False)
        # 设置取消按钮不能按
        self.reset.setEnabled(True)

    def to_six(self, y, posistion):
        y = round(y*100, 1)
        return str("%.1f"%y)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    player = FaceSRMainWindow()
    player.show()
    sys.exit(app.exec_())
