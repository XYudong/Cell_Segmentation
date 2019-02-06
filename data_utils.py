import numpy as np
import os
import cv2
import glob
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class DataPreparer:
    CROP_SIZE = 128
    MARGIN = 30         # (128 - 68) / 2
    SPLIT_RATE = 0.2

    def __init__(self, im_path, mask_path, crop_num=200, batch_size=32):
        self.im_path = im_path
        self.mask_path = mask_path
        self.crop_num = crop_num
        self.imgs = []
        self.masks = []
        self.edges = []
        self.img_list = []      # for file names
        self.mask_list = []
        self.batch_size = batch_size
        self.num_train = None
        self.num_val = None

    def load_img_mask(self):
        """load images and masks from disk and get edges"""
        self.img_list = glob.glob(os.path.join(self.im_path, '*.tif'))
        self.mask_list = glob.glob(os.path.join(self.mask_path, '*.png'))
        assert len(self.img_list) == len(self.mask_list), 'inconsistent number of imgs and masks'
        return

    def get_edge(self, mask):
        """detect edges from input mask"""
        mask[mask > 0.5] = 255
        edg = cv2.Canny(mask, 50, 100)
        kernel = np.ones((3, 3), np.uint8)
        edg = cv2.dilate(edg, kernel, iterations=3)     # needs tuning
        return edg

    def crop_on_loc(self, inp, rows, cols):
        """crop given input at rows and cols, cropping size: CROP_SIZE x CROP_SIZE"""
        out = []
        offset0 = int(self.CROP_SIZE / 2)
        offset1 = self.CROP_SIZE - offset0
        for row, col in zip(rows, cols):
            out.append(inp[row - offset0:row + offset1, col - offset0:col + offset1])
        out = np.stack(out, axis=0)
        return out

    def sample_loc(self, edge, number, on_edge=True):
        if on_edge:
            loc = np.where(edge > 0)      # a tuple of two arrays, represent indices of row and col respectively
        else:
            loc = np.where(edge < 1)
        sample_idx = np.random.randint(0, high=len(loc[0]), size=number)
        return loc[0][sample_idx], loc[1][sample_idx]

    def crop_all(self):
        edge_ratio = 0.7
        edge_num = int(self.crop_num * edge_ratio)
        back_num = self.crop_num - edge_num
        pad_width = int(np.ceil(self.CROP_SIZE / 2))
        moving_sum = []

        for img_name, mask_name in zip(self.img_list, self.mask_list):
            img = cv2.imread(img_name, 0).astype('float32')
            mask = cv2.imread(mask_name, 0)
            edge = self.get_edge(mask)
            moving_sum.append(img)

            row_p, col_p = self.sample_loc(edge, edge_num, True)
            row_n, col_n = self.sample_loc(edge, back_num, False)
            rows = np.hstack((row_p, row_n)) + pad_width
            cols = np.hstack((col_p, col_n)) + pad_width

            img = np.lib.pad(img, pad_width, 'symmetric')
            mask = np.lib.pad(mask, pad_width, 'symmetric')
            edge = np.lib.pad(edge, pad_width, 'symmetric')

            self.imgs.append(self.crop_on_loc(img, rows, cols))
            self.masks.append(self.crop_on_loc(mask, rows, cols))
            self.edges.append(self.crop_on_loc(edge, rows, cols))

        # normalization
        img_mean, img_std = self.get_mean(moving_sum), self.get_std(moving_sum)
        np.savez(self.im_path + '/train_mean_std.npz', mean=img_mean, std=img_std)
        self.imgs = (self.imgs - img_mean) / img_std

        self.imgs = np.concatenate(self.imgs, axis=0)
        self.masks = self.crop_margin(self.masks)
        self.edges = self.crop_margin(self.edges)
        return

    def crop_margin(self, inp):
        """cut out given MARGIN from the inp"""
        inp = np.concatenate(inp, axis=0).astype('float32')
        return inp[:, self.MARGIN:-self.MARGIN, self.MARGIN:-self.MARGIN]

    def binarize_mask_edge(self):
        # need the operand to be float type
        self.masks /= 255.
        self.edges /= 255.
        return

    def get_generator(self):
        """get generators for training set and validation set"""
        generator = ImageDataGenerator(rotation_range=30,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='reflect')
        # preprocessing
        self.binarize_mask_edge()

        self.imgs = self.add_axis(self.imgs, repeat=True)
        self.masks = self.add_axis(self.masks)
        self.edges = self.add_axis(self.edges)
        seed = 66
        # split data
        imgs_tr, imgs_val, masks_tr, masks_val, edges_tr, edges_val\
            = train_test_split(self.imgs, self.masks, self.edges,
                               test_size=self.SPLIT_RATE, random_state=seed)
        self.num_train = len(imgs_tr)
        self.num_val = len(imgs_val)

        # feed generator with the corresponding data
        gene_img = generator.flow(imgs_tr, batch_size=self.batch_size, seed=seed)
        gene_mask = generator.flow(masks_tr, batch_size=self.batch_size, seed=seed)
        gene_edge = generator.flow(edges_tr, batch_size=self.batch_size, seed=seed)
        out_gene = zip(gene_mask, gene_edge)
        train_generator = zip(gene_img, out_gene)

        gene_img = generator.flow(imgs_val, batch_size=self.batch_size, seed=seed)
        gene_mask = generator.flow(masks_val, batch_size=self.batch_size, seed=seed)
        gene_edge = generator.flow(edges_val, batch_size=self.batch_size, seed=seed)
        out_gene = zip(gene_mask, gene_edge)
        val_generator = zip(gene_img, out_gene)

        return train_generator, val_generator

    def add_axis(self, img, repeat=False):
        img = img[..., np.newaxis]
        return img if not repeat else img.repeat(3, axis=-1)

    def get_mean(self, imgs):
        return np.mean(imgs)

    def get_std(self, imgs):
        return np.std(imgs)

    def save_mean_std(self):
        pass

    def main(self):
        self.load_img_mask()
        self.crop_all()
        gene = self.get_generator()
        return gene















