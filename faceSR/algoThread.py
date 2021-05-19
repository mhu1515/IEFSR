import sys
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap
from models.LR_encoder import *
from PIL import Image
from models.IEFSR import generator
from torchvision import transforms
from utils import *
import Debuger
import keras
from models import coarse_LR as Network
from models import coarse_LR_utils as utls

curr_path = os.path.abspath(os.path.dirname(__file__))

class algoThread(QThread):
    trigger = pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        super(algoThread, self).__init__(parent)
        self.psnrVal = 0
        self.ssimVal = 0
        self.x_image = None

    def setScale(self, scale):
        self.scale = scale

    def setInput(self, img):
        self.x_image = img

    def save_image(self):
        filename = os.path.join(curr_path, 'results', 'predict' + '.jpg')
        img = self.output[0].clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0)
        img = img.astype("uint8")
        img = img[:, :, (2, 1, 0)]
        img = Image.fromarray(img)
        img.save(filename)

    def cal_index(self):
        # Debuger.debug("计算指标")

        img = self.output[0].clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")

        self.entropy = entropy_My(img)   # 信息熵
        self.light = avgLightImg(img)   #亮度
        self.gradient = avgGradient(img)  # 梯度
        self.std = stdImg(img)   # 对比度

    def run(self):
        '''
        运行算法
        :param x_image: Image
        :return:
        '''

        Debuger.debug("运行算法")

        coarse_LR = Network.build_coarse_LR((None, None, 3))
        net_path = os.path.join(curr_path, 'pretrained_models', '200_dark_base.h5')
        coarse_LR.load_weights(net_path)
        opt = keras.optimizers.Adam(lr=2 * 1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        coarse_LR.compile(loss='mse', optimizer=opt)

        flag = 0
        lowpercent = 5
        highpercent = 95
        maxrange = 8 / 10.
        hsvgamma = 8 / 10.

        img_A = np.array(self.x_image, dtype='float64')/ 255.
        b, g, r = cv2.split(img_A)
        img_A = cv2.merge([r, g, b])
        img_A = img_A[np.newaxis, :]

        out_pred = mbllen.predict(img_A)
        fake_B = out_pred[0, :, :, :3]
        fake_B_o = fake_B

        gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
        percent_max = sum(sum(gray_fake_B >= maxrange)) / sum(sum(gray_fake_B <= 1.0))
        # print(percent_max)
        max_value = np.percentile(gray_fake_B[:], highpercent)
        if percent_max < (100 - highpercent) / 100.:
            scale = maxrange / max_value
            fake_B = fake_B * scale
            fake_B = np.minimum(fake_B, 1.0)

        gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
        sub_value = np.percentile(gray_fake_B[:], lowpercent)
        fake_B = (fake_B - sub_value) * (1. / (1 - sub_value))

        imgHSV = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(imgHSV)
        S = np.power(S, hsvgamma)
        imgHSV = cv2.merge([H, S, V])
        fake_B = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
        fake_B = np.minimum(fake_B, 1.0)

        if flag:
            outputs = np.concatenate([img_A[0, :, :, :], fake_B_o, fake_B], axis=1)
        else:
            outputs = fake_B

        # scipy.misc.toimage(outputs * 255, high=255, low=0, cmin=0, cmax=255).save(img_name)
        outputs = np.minimum(outputs, 1.0)
        outputs = np.maximum(outputs, 0.0)

        device = torch.device("cuda")
        x_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])

        r, g, b = cv2.split(outputs * 255)
        img_rgb = cv2.merge([r, g, b])
        # img_bgr = cv2.merge([b, g, r])
        # img_bgr = img_bgr.astype(np.uint8)
        img_rgb = img_rgb.astype(np.uint8)
        # cv2.imwrite("./results/enhance_bgr.jpg", img_bgr)
        cv2.imwrite("./results/enhance_rgb.jpg", img_rgb)
        print('enhance_rgb')
        # img_pil = Image.open("./results/enhance_rgb.jpg")
        self.x_image = x_transform(img_rgb)
        self.x_image = self.x_image.unsqueeze(0).to(device)

        with torch.no_grad():
            save_enc_model = os.path.join(curr_path, 'pretrained_models', 'enc.pth')
            save_gen_model = os.path.join(curr_path, 'pretrained_models', 'gen.pth')
            encoder = gaussian_resnet_encoder(image_size=32).to(device)
            G = generator(scale=self.scale).to(device)
            state_enc_dict = torch.load(save_enc_model)
            state_gen_dict = torch.load(save_gen_model)
            encoder.load_state_dict(state_enc_dict)
            G.load_state_dict(state_gen_dict)
            encoder.eval()
            G.eval()
            z, mu, logvar = encoder(self.x_image)
            self.output = G(self.x_image, mu)
            self.x_image = self.x_image.cpu()
            self.output = self.output.cpu()
        self.x_image[0] *= 255
        self.output[0] *= 255

        self.save_image()
        self.cal_index()
        img = self.output[0].clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        img = img[:, :, (2, 1, 0)]
        img = Image.fromarray(img).toqpixmap()
        # 发射完成信号
        self.trigger.emit(img)
