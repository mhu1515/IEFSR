import os
import numpy as np
from PIL import Image, ImageDraw
import torch.utils.data as data


# CELEB_ROOT = '/home/gwnam/datasets/CelebAMask-HQ/'
CELEB_ROOT = './datasets/'


class CelebDataset(data.Dataset):

    def __init__(self, root=CELEB_ROOT, mode='train', use_hmaps=False, use_pmaps=True,
                size_lr = (16, 16), size_hr = (128,128), size_maps = (64, 64)):
                # size_lr = (32, 32), size_hr = (128, 128), size_maps = (64, 64)):
        self.root = root
        self.mode = mode
        self.use_hmaps = use_hmaps
        self.use_pmaps = use_pmaps
        self.size_lr = size_lr
        self.size_hr = size_hr
        self.size_maps = size_maps
        self.img_info = list()
        self.list_pmaps = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
                           'l_ear', 'r_ear', 'nose', 'mouth', 'u_lip', 'l_lip']
        # self.list_pmaps = ['cloth','hair','neck','skin','ear_r','l_brow', 'r_brow', 'l_eye', 'r_eye',
        #                    'l_ear', 'r_ear', 'nose', 'mouth', 'u_lip', 'l_lip']
        # self.list_pmaps = ['skin', 'lbrow', 'rbrow', 'leye', 'reye',
        #                    'lear', 'rear', 'nose', 'mouth', 'ulip', 'llip']
        if self.mode == 'train':
            img_list = os.path.join(self.root, 'train_fileList.txt')
            # img_list = os.path.join(self.root, 'fileList_trn_18000.txt')
        elif self.mode == 'test':
            # img_list = os.path.join(self.root, 'test_lowlight.txt')
            # img_list = os.path.join(self.root, 'fileList_tst_100.txt')
            img_list = os.path.join(self.root,'test_fileList.txt')
        else:
            print('not implemented')
            exit()
        # image file name without type
        with open(img_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                fname = int(line.strip()[:-4])
                # fname = int(line.strip()[:-11])
                self.img_info.append({
                    'image': fname,
                })

    def num_channels(self):
        return 0, len(self.list_pmaps)
        # return 136, len(self.list_pmaps)
    def _get_hmaps(self, lmark):
        print('not implemented')
        exit()
        # hmaps = np.zeros((self.size_maps[0], self.size_maps[1], 136), dtype=np.uint8)
        # for i, p in enumerate(lmark):
        #     hmaps[:, :, i] = self._gaussian_k(p[0], p[1], sigma=3,
        #                                       width=self.size_maps[0],
        #                                       height=self.size_maps[1])
        # return hmaps

    def _get_pmaps(self, index):
        pmaps = np.zeros((self.size_maps[0], self.size_maps[1], len(self.list_pmaps)), dtype=np.uint8)
        face_idx = self.img_info[index]['image']
        for i, tail in enumerate(self.list_pmaps):
            # '%05d_' % face_idx 五位数字格式化文件名
            anno_img = os.path.join(self.root, 'CelebAMask-HQ-mask-anno/merged/', '%05d_' % face_idx + tail + '.png')
            # anno_img = os.path.join(self.root, 'Parsing_Maps/', str(face_idx) +'_'+ tail + '.png')
            if not os.path.exists(anno_img):
                # keep black
                print('000')
                continue
            else:
                print('111')
            anno_img = Image.open(anno_img).convert('L')
            # 三次样条插值 resize 64 × 64
            anno_img = anno_img.resize(self.size_maps, Image.BICUBIC)
            pmaps[:,:,i] = np.array(anno_img)
        return pmaps

    def __getitem__(self, index):

        # it will be returned
        image_lr = None
        image_hr = None
        hmaps = np.array([-1])
        pmaps = np.array([-1])
    
        # load infos
        face_idx = self.img_info[index]['image']
        # face_img = Image.open(os.path.join(self.root, ''
        #                                               'CelebA-HQ-img', str(face_idx) + '.jpg')).convert('RGB')
        face_lr_img = Image.open(os.path.join(self.root,'./CelebA-HQ-dark5/', str(face_idx) + '.jpg')).convert('RGB')
        face_hr_img = Image.open(os.path.join(self.root,'CelebA-HQ-img', str(face_idx) + '.jpg')).convert('RGB')
        # resize
        # face_img 1024*1024 image_lr 16 * 16
        image_lr = face_lr_img.resize(self.size_lr, Image.BICUBIC)
        image_lr = image_lr.resize(self.size_hr, Image.BICUBIC)
        # image_lr = image_lr.resize(self.size_hr, Image.BICUBIC)
        # image_lr 128 * 128
        # image_lr = image_lr.resize(self.size_hr, Image.BICUBIC)
        # image_hr 128 * 128
        # image_lr = face_img.resize(self.size_hr, Image.BICUBIC)
        # image_hr = face_img.resize(self.size_hr, Image.BICUBIC)
        image_hr = face_hr_img.resize(self.size_hr, Image.BICUBIC)

        # hmaps & pmaps
        if self.use_hmaps:
            # hmaps = self._get_hmaps(lmark)
            hmaps = self._get_hmaps(index)
        if self.use_pmaps:
            pmaps = self._get_pmaps(index)

        return np.array(image_lr), np.array(image_hr), hmaps, pmaps

    def __len__(self):
        return len(self.img_info)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage.measure import compare_psnr
    from skimage.measure import compare_ssim

    dset = CelebDataset(mode='test')
    for idx, (image_lr, image_hr, hmaps, pmaps) in enumerate(dset, start=1):

        if idx == 2:
            break

        print()
        print('psnr:', compare_psnr(image_lr, image_hr))
        print('ssim:', compare_ssim(image_lr, image_hr, multichannel=True))

        # lr & hr images
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(image_lr)
        fig.add_subplot(1, 2, 2)
        plt.imshow(image_hr)
        plt.show()

        # pmaps
        fig = plt.figure()
        for i in range(11):
            _img = Image.fromarray(pmaps[:,:,i])
            fig.add_subplot(4, 3, i+1)
            plt.imshow(_img, cmap='gray', vmin=0, vmax=255)
        plt.show()
