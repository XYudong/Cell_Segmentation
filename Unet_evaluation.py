from evaluation_utils import *
import numpy as np
import cv2
from skimage import color
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model


# initialization
im_path = './DataSet_label/FAK_N1/test'
mask_path = './DataSet_label/FAK_N1/test_mask'
train_stats_path = 'DataSet_label/FAK_N1/img/train_mean_std.npz'
batch_size = 16


def get_model(model_name):
    """build a model from saved model with new input shape"""
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
    imgs, masks, edges = test_preparer.get_imgs()

    model = get_model(model_name)
    model.compile(optimizer='adam',
                  loss=['binary_crossentropy', 'binary_crossentropy'],  # mask, edge
                  metrics=[dice_coef],
                  loss_weights=[0.9999, 0.0001])

    outputs = model.test_on_batch(imgs, [masks, edges])
    print('loss and coef: ', outputs)

    return


def predict_mask_v2(model_name, save_path=None):
    """input the images in whole"""
    print('pre-trained model: ', model_name)
    test_preparer = SegPreparer(im_path, train_stats_path)
    imgs = test_preparer.get_imgs()

    model = get_model(model_name)
    pred_mask, pred_edge = model.predict(imgs)      # two lists

    pred_mask = postprocess(pred_mask[0])
    pred_edge = postprocess(pred_edge[0])

    if save_path is not None:
        save_img(pred_mask, save_path + '/pred_mask_D3.png')
        save_img(pred_edge, save_path + '/pred_edge_D3.png')
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


def postprocess(img):
    assert np.max(img) <= 1, 'input img is not gray scale image'
    img *= 255

    MARGIN = 30  # because of Cropping2D layer
    img = np.pad(img[:, :, 0], MARGIN, 'symmetric').astype('uint8')
    img = np.pad(img, ((0, 0), (0, 8)), 'symmetric')  # because of new input shape
    return img


def save_img(img, file_path):
    cv2.imwrite(file_path, img)
    print('img saved: ', file_path)
    return


def overlay_img_mask(img_path, mask_path):
    """overlay raw gray-scale image with predicted mask"""
    img = cv2.imread(img_path, 0)
    img = color.gray2rgb(img)
    plt.figure()
    plt.imshow(img)

    mask = cv2.imread(mask_path, 0)
    regions = mask > 0

    channel_multipler = [0, 1, 1.5]
    img = img.astype('float32')
    img[regions, :] *= channel_multipler

    plt.figure()
    plt.imshow(img.astype('uint8'))

    plt.show()


if __name__ == '__main__':
    model_name = 'vUnet_FAK_N1_08.hdf5'
    # evaluate_model(model_name)
    # predict_mask_v2(model_name, save_path=)

    # colorize gray-scale image
    img_path = 'DataSet_label/FAK_N1/test_mask/C3_01_2_1_GFP_001.png'
    pred_mask_path = 'results/predict/FAK_N1/model_08/pred_mask_C3.png'
    overlay_img_mask(img_path, pred_mask_path)






