import numpy as np
from os import path
import glob
import cv2
import matplotlib.pyplot as plt

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


def fn():
    a = 'aaa'
    b = ['bbb', 'ccc', 'ddd']
    print(b[0:2])


if __name__ == '__main__':
    # extract_fea()
    fn()




