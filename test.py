import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import heapq
# import tensorflow as tf
from data_utils import DataPreparer

# read_msk('DataSet_label/FAK_N1/img/C1_01_1_1_Bright Field_001.tif')

# r_path = './DataSet_label/FAK_N1/img'
# files = os.path.join(r_path, '*.tif')
# img_list = glob.glob(files)
#
# print(files)
#
# print(img_list)


# def get_dataset(self):
#     """turn data into tf.data.Dataset"""
#     input_set = tf.data.Dataset.from_tensor_slices(self.imgs)
#     output_set = tf.data.Dataset.from_tensor_slices((self.masks, self.edges))
#     dataset = tf.data.Dataset.zip((input_set, output_set))
#     return dataset.batch(self.batch_size).repeat().prefetch(1)
#
#
# def _parse_func(self, img, mask_edge):
#     imgs = self.preprocess(img)
#     masks = self.preprocess(mask_edge[0])
#     edges = self.preprocess(mask_edge[1])
#     return imgs, (masks, edges)
#
#
# def preprocess(self, inp):
#     out = [inp]
#     out.append(tf.image.flip_left_right(inp))
#     out.append(tf.image.random_flip_left_right(inp))
#     out.append(tf.image.rot90(inp, k=1))
#     out.append(tf.image.rot90(inp, k=2))
#     return out


def extract_fea():
    mask = cv2.imread('DataSet_label/FAK_N3/GFP_MASK_PNG/D2_01_2_1_GFP_001.png', 0)
    contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.imread('DataSet_label/FAK_N3/Bright_Field/D2_01_1_1_Bright Field_001.tif', 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    areas = []
    for cnt in contours:
        areas.append(cv2.contourArea(cnt))
    idx = np.argsort(areas)[-2:]

    cnt = contours[idx[0]]
    img_contour = cv2.drawContours(img, [cnt], 0, (0, 250, 0), 3)
    cnt = contours[idx[1]]
    img_contour = cv2.drawContours(img_contour, [cnt], 0, (0, 0, 250), 3)

    plt.figure()
    plt.imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
    plt.show()


def test_fn():
    img0 = cv2.imread('DataSet_label/FAK_N1/train/C1_01_1_1_Bright Field_001.tif', 0)
    img1 = cv2.imread('DataSet_label/Human_Muscle_PF573228/FAK_N4_Gray/0001.png', 0)

    plt.figure()
    plt.imshow(img0, 'gray')

    plt.figure()
    plt.imshow(img1, 'gray')

    img3 = img1.copy()
    img3[img3 < 130] = 30
    plt.figure()
    plt.imshow(img3, 'gray')

    plt.show()


if __name__ == '__main__':
    # extract_fea()
    test_fn()
















