import numpy as np
import os
import cv2
import glob
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class DataPreparer:
    CROP_SIZE = 128
    MARGIN = 30
    SPLIT_RATE = 0.2

    def __init__(self, im_path, mask_path, crop_num=200, batch_size=32):
        self.im_path = im_path
        self.mask_path = mask_path
        self.crop_num = crop_num
        self.imgs = []
        self.masks = []
        self.edges = []
        self.batch_size = batch_size
        self.num_train = 0
        self.num_val = 0

    def load_img_mask(self):
        """load images and masks from disk and get edges"""
        img_list = glob.glob(os.path.join(self.im_path, '*.tif'))
        mask_list = glob.glob(os.path.join(self.mask_path, '*.png'))

        for i in range(len(img_list)):
            self.imgs.append(cv2.imread(img_list[i], 0))
            self.masks.append(cv2.imread(mask_list[i], 0))
            self.get_edge(self.masks[-1])
        assert len(self.imgs) == len(self.edges) == len(self.masks), 'inconsistent number of inputs and outputs'
        return

    def get_edge(self, mask):
        """detect edges from input mask"""
        mask[mask > 0.5] = 255
        edg = cv2.Canny(mask, 50, 100)
        kernel = np.ones((3, 3), np.uint8)
        edg = cv2.dilate(edg, kernel, iterations=3)     # needs tuning
        self.edges.append(edg)
        return

    def crop_on_loc(self, inp, rows, cols):
        """crop given input according to rows and cols, cropping size is related to CROP_SIZE"""
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

        imgs, masks, edges = [], [], []
        for img, mask, edge in zip(self.imgs, self.masks, self.edges):
            row_p, col_p = self.sample_loc(edge, edge_num, True)
            row_n, col_n = self.sample_loc(edge, back_num, False)
            rows = np.hstack((row_p, row_n)) + pad_width
            cols = np.hstack((col_p, col_n)) + pad_width

            img = np.lib.pad(img, pad_width, 'symmetric')
            mask = np.lib.pad(mask, pad_width, 'symmetric')
            edge = np.lib.pad(edge, pad_width, 'symmetric')

            imgs.append(self.crop_on_loc(img, rows, cols))
            masks.append(self.crop_on_loc(mask, rows, cols))
            edges.append(self.crop_on_loc(edge, rows, cols))

        # self.imgs = np.stack(imgs, axis=2).reshape((-1, self.CROP_SIZE, self.CROP_SIZE)).astype('float32')
        self.imgs = np.concatenate(imgs, axis=0).astype('float32')
        self.masks = self.crop_margin(masks)
        self.edges = self.crop_margin(edges)
        return

    def crop_margin(self, inp):
        """cut out given MARGIN from the inp"""
        inp = np.concatenate(inp, axis=0).astype('float32')
        return inp[:, self.MARGIN:-self.MARGIN, self.MARGIN:-self.MARGIN]

    # def get_dataset(self):
    #     """turn data into tf.data.Dataset"""
    #     input_set = tf.data.Dataset.from_tensor_slices(self.imgs)
    #     output_set = tf.data.Dataset.from_tensor_slices((self.masks, self.edges))
    #     dataset = tf.data.Dataset.zip((input_set, output_set))
    #     return dataset.batch(self.batch_size).repeat().prefetch(1)

    def binarize_mask_edge(self):
        # need the operand to be float type
        self.masks /= 255.
        self.edges /= 255.
        return

    def get_generator(self):
        generator = ImageDataGenerator(rotation_range=90,
                                       zoom_range=0.3,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='reflect')
        # preprocessing
        self.binarize_mask_edge()
        self.imgs = (self.imgs - self.get_mean()) / self.get_std()   # zero-mean or normalization?
        self.imgs = self.add_axis(self.imgs).repeat(3, axis=-1)
        self.masks = self.add_axis(self.masks)
        self.edges = self.add_axis(self.edges)
        seed = 66
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

    def add_axis(self, img):
        return img[..., np.newaxis]

    def get_mean(self):
        return np.mean(self.imgs)

    def get_std(self):
        return np.std(self.imgs)

    # def get_num(self):
    #     return int(len(self.imgs) * self.SPLIT_RATE)

    def main(self):
        self.load_img_mask()
        self.crop_all()
        gene = self.get_generator()
        return gene
















