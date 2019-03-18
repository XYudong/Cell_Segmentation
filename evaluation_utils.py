import numpy as np
from os import listdir, path
import cv2
import glob
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
"""prepare test images before going through Unet segmentation model"""


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def get_edge(mask, iter=3):
    """detect edges from input mask"""
    # mask[mask < 30] = 0           # can filter out some small values
    edg = cv2.Canny(mask, 80, 150)
    kernel = np.ones((3, 3), np.uint8)
    edg = cv2.dilate(edg, kernel, iterations=iter)     # needs tuning
    return edg


def get_filename(file_path):
    """get only file names in the input directory"""
    files = [f for f in listdir(file_path) if path.isfile((path.join(file_path, f)))]
    return sorted(files)


class SegPreparer:
    INPUT_SIZE = 128    # input size of our Unet
    OUTPUT_SIZE = 68

    def __init__(self, im_path, train_stats_path, mask_path=None):
        self.im_path = im_path
        self._mask_path = mask_path
        self.stats_path = train_stats_path
        self.imgs = []
        self.masks = []
        self.edges = []
        self.img_list = []
        self.mask_list = []
        self.num_x = None   # number of crops in x direction(horizontal)
        self.num_y = None
        self.margin_x = None
        self.margin_y = None
        self.shape = None   # shape of an original image
        self.num_test = None    # number of test images

    def load_test_set(self):
        """load test set image names"""
        self.img_list = sorted(glob.glob(path.join(self.im_path, '*.png')))
        self.num_test = len(self.img_list)
        assert self.num_test > 0, "empty test set: " + str(self.num_test)
        print(self.img_list)
        print(self.num_test, ' images')

        if self._mask_path is not None:
            self.mask_list = sorted(glob.glob(path.join(self._mask_path, '*.png')))
            assert len(self.mask_list) == self.num_test, "different number of test images and masks"
        return

    def init_(self):
        img = cv2.imread(self.img_list[0], 0)
        self.shape = img.shape
        self.num_x = int(np.ceil(self.shape[1] / self.OUTPUT_SIZE))
        self.num_y = int(np.ceil(self.shape[0] / self.OUTPUT_SIZE))
        self.margin_x = self.num_x * self.OUTPUT_SIZE - self.shape[1]
        self.margin_y = self.num_y * self.OUTPUT_SIZE - self.shape[0]
        return

    def get_stats(self, file_path):
        npzfile = np.load(file_path)
        x_mean, x_std = npzfile['mean'], npzfile['std']
        return x_mean, x_std

    def crop_all(self):
        self.init_()
        mean, std = self.get_stats(self.stats_path)     # images stats from training set
        pad_width = int(np.ceil((self.INPUT_SIZE - self.OUTPUT_SIZE) / 2))
        if self._mask_path is not None:
            for img_name, mask_name in zip(self.img_list, self.mask_list):
                # image
                img = (cv2.imread(img_name, 0) - mean) / std
                img = self.pad_each(img, pad_width)
                img = self.crop_each(img, self.INPUT_SIZE)
                self.imgs.append(img[..., np.newaxis].repeat(3, axis=-1))

                # mask and edge
                mask = cv2.imread(mask_name, 0)
                edge = get_edge(mask)

                mask = self.pad_each(mask, 0)
                mask = self.crop_each(mask, self.OUTPUT_SIZE)
                self.masks.append(mask[..., np.newaxis])

                edge = self.pad_each(edge, 0)
                edge = self.crop_each(edge, self.OUTPUT_SIZE)
                self.edges.append(edge[..., np.newaxis])
        else:
            for img_name in self.img_list:
                img = (cv2.imread(img_name, 0) - mean) / std
                img = self.pad_each(img, pad_width)
                img = self.crop_each(img, self.INPUT_SIZE)
                self.imgs.append(img[..., np.newaxis].repeat(3, axis=-1))
        return

    def pad_each(self, inp, extra_pad_width):
        inp = np.pad(inp, ((extra_pad_width, extra_pad_width + self.margin_y),
                           (extra_pad_width, extra_pad_width + self.margin_x)),
                     'symmetric')
        return inp

    def crop_each(self, inp, size):
        crops = []
        for row in range(self.num_y):       # the order matters
            for col in range(self.num_x):
                row_start = row * self.OUTPUT_SIZE
                col_start = col * self.OUTPUT_SIZE
                crops.append(inp[row_start:row_start + size,
                             col_start:col_start + size])
        return np.stack(crops, axis=0)      # (N, 128, 128) or (N, 68, 68)

    # def toImages(self, imgs, file_name):
        # """save masks or edges from Unet to images"""
        # # for f in range(self.num_test):
        # out = np.zeros((self.num_y * self.OUTPUT_SIZE, self.num_x * self.OUTPUT_SIZE))
        # imgs *= 255
        # for idx, img_crop in enumerate(imgs):
        #     col = idx % self.num_x
        #     row = np.floor(idx / self.num_x)
        #     out[int(row * self.OUTPUT_SIZE): int((row + 1) * self.OUTPUT_SIZE),
        #         int(col * self.OUTPUT_SIZE): int((col + 1) * self.OUTPUT_SIZE)]\
        #         = img_crop.reshape((self.OUTPUT_SIZE, self.OUTPUT_SIZE)).astype('uint8')
        #
        # cv2.imwrite(file_name, out[:self.shape[0], :self.shape[1]])
        # print('image saved')
        # return

    def get_crops(self):
        """get crops of an image"""
        self.load_test_set()
        self.crop_all()
        if self._mask_path is None:
            self.imgs = np.concatenate(self.imgs, axis=0)
            return self.imgs        # for prediction
        else:
            self.imgs = np.concatenate(self.imgs, axis=0)
            self.masks = np.concatenate(self.masks, axis=0)
            self.edges = np.concatenate(self.edges, axis=0)
            return self.imgs, self.masks, self.edges     # arrays, for evaluation

    def get_imgs(self):
        """get each image in whole instead of crops"""
        self.load_test_set()
        mean, std = self.get_stats(self.stats_path)     # images stats from training set
        imgs = []
        if self._mask_path is None:
            for img_name in self.img_list:
                img = cv2.resize(cv2.imread(img_name, 0), (1128, 832))
                img = (img - mean) / std
                imgs.append(img[:, 0:-8, np.newaxis].repeat(3, axis=-1))
            return np.stack(imgs, axis=0)   # (N, 832, 1128-8, 3)
        else:
            masks = []
            edges = []
            for img_name, mask_name in zip(self.img_list, self.mask_list):
                img = (cv2.imread(img_name, 0) - mean) / std
                mask = cv2.imread(mask_name, 0)
                edge = get_edge(mask) / 255
                mask = mask / 255                   # model target should be binary mask!

                imgs.append(img[:, 0:-8, np.newaxis].repeat(3, axis=-1))    # model input
                masks.append(mask[30:-30, 30:-38, np.newaxis])              # model output
                edges.append(edge[30:-30, 30:-38, np.newaxis])
            return np.stack(imgs, axis=0), np.stack(masks, axis=0), np.stack(edges, axis=0)
