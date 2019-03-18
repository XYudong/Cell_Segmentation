import numpy as np
from os import path
import glob
import cv2
import matplotlib.pyplot as plt
import heapq
import tensorflow as tf

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


def test_fn():
    # img0 = cv2.imread('DataSet_label/Human_Muscle_PF573228/FAK_N4_Gray/train/img/0001.png', 0)
    # img1 = cv2.imread('DataSet_label/Human_Muscle_PF573228/FAK_N4_Gray/train/mask/0001_mask.png', 0)
    # img2 = cv2.imread('test_img.png', 0)
    #
    # # img1[img1 > 0] = 255
    # # cv2.imwrite('test_img.png', img1)
    #
    # plt.figure()
    # plt.imshow(img0)
    #
    # plt.figure()
    # plt.imshow(img1)
    #
    # # img3 = img1.copy()
    # # img3[img3 > 1] = 30
    # # plt.figure()
    # # plt.imshow(img3, 'gray')
    #
    # plt.show()

    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))


if __name__ == '__main__':
    # extract_fea()
    test_fn()
















