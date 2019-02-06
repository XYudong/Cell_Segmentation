from evaluation_utils import *
import numpy as np
# import os
import cv2
import tensorflow as tf
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K


im_path = './DataSet_label/FAK_N1/test'
mask_path = './DataSet_label/FAK_N1/test_mask'
train_stats_path = 'DataSet_label/FAK_N1/img/train_mean_std.npz'
batch_size = 16


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def get_model(model_name):
    model = tf.keras.models.load_model('./results/model/' + model_name,
                                       custom_objects={'dice_loss': dice_loss,
                                                       'dice_coef': dice_coef},
                                       compile=True)
    input_layer = Input(shape=(832, 1120, 3), name='new_input')
    model_outputs = model(input_layer)
    model = Model(input_layer, model_outputs)
    model.summary()
    return model


def evaluate_model(model_name):
    print('pre-trained model: ', model_name)
    test_preparer = SegPreparer(im_path, train_stats_path, mask_path)
    imgs, masks, edges = test_preparer.get_crops()

    model = tf.keras.models.load_model('./results/model/' + model_name,
                                       custom_objects={'dice_loss': dice_loss,
                                                       'dice_coef': dice_coef},
                                       compile=True)

    outputs = model.evaluate(imgs, [masks, edges], batch_size)         # edges

    print('loss and coef: ', outputs)

    return


def predict_mask_v2(model_name):
    """input the images in whole"""
    print('pre-trained model: ', model_name)
    test_preparer = SegPreparer(im_path, train_stats_path)
    imgs = test_preparer.get_imgs()

    model = get_model(model_name)
    pred_mask, pred_edge = model.predict(imgs)

    save_img(pred_mask[0], './results/predict/pred_mask_C3.png')
    save_img(pred_edge[0], './results/predict/pred_edge_C3.png')
    return


def predict_mask(model_name):
    """input the images in crops"""
    print('pre-trained model: ', model_name)
    test_preparer = SegPreparer(im_path, train_stats_path)
    imgs = test_preparer.get_crops()

    model = tf.keras.models.load_model('./results/model/' + model_name,
                                       custom_objects={'dice_loss': dice_loss,
                                                       'dice_coef': dice_coef},
                                       compile=True)
    pred_mask, pred_edge = model.predict(imgs)

    test_preparer.toImages(pred_mask, './results/predict/pred_mask_C3.png')
    test_preparer.toImages(pred_edge, './results/predict/pred_edge_C3.png')

    return


def save_img(img, file_path):
    assert np.max(img) <= 1, 'input img is not gray scale image'
    img *= 255
    img = img.reshape(img.shape[0:2]).astype('uint8')
    img = np.pad(img, ((0, 0), (0, 8)), 'symmetric')
    cv2.imwrite(file_path, img)
    print('img saved: ', file_path)


if __name__ == '__main__':
    model_name = 'vUnet_FAK_N1_05.hdf5'
    # evaluate_model(model_name)
    predict_mask_v2(model_name)

    # get_model(model_name)






