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
    # img0 = cv2.imread('results/predict/HM_FAK_N6/N4_model_03/predMask/predMask_0029.png', 0)

    img = np.zeros((100, 100)).astype(np.uint8)
    cv2.circle(img, (50, 50), 20, 1, thickness=-1)
    cv2.circle(img, (50, 50), 10, 0, thickness=-1)

    cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts]
    print(areas)

    print(np.pi * 10**2)
    print(np.pi * 20**2)

    plt.figure()
    plt.imshow(img, 'gray')

    plt.show()


if __name__ == '__main__':
    # extract_fea()
    # test_fn()
    a = [-2, 2]
    print(np.linalg.norm(a))







