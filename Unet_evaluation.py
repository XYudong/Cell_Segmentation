from evaluation_utils import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model


THRESHOLD = 100     # threshold for predicted mask


def get_model(model_name):
    """build a model from saved model with the new input shape"""
    model = tf.keras.models.load_model('./results/model/' + model_name,
                                       custom_objects={'dice_loss': dice_loss,
                                                       'dice_coef': dice_coef},
                                       compile=True)
    input_layer = Input(shape=(832, 1120, 3), name='new_input')
    model_outputs = model(input_layer)
    model = Model(input_layer, model_outputs)
    model.summary()
    return model


def evaluate_model(model_name, out_path=None):
    print('pre-trained model: ', model_name)
    test_preparer = SegPreparer(raw_im_path, train_stats_path, gt_mask_path)
    imgs, masks, edges = test_preparer.get_imgs()

    model = get_model(model_name)
    model.compile(optimizer='adam',
                  loss=['binary_crossentropy', 'binary_crossentropy'],  # mask, edge
                  metrics=[dice_coef],
                  loss_weights=[0.999, 0.001])
    outputs = []
    for img, mask, edge in zip(imgs, masks, edges):
        output = model.evaluate(img[np.newaxis, ...], [mask[np.newaxis, ...], edge[np.newaxis, ...]])
        print('sub loss and coef: ', output)
        outputs.append(output)

    outputs = np.mean(outputs, axis=0)
    print('\naverage loss and coef: ', outputs)
    if out_path is not None:
        loss, mask_loss, edge_loss, mask_coef, edge_coef = outputs
        file_path = out_path + '/eval_' + model_name[6:-5]
        np.savez(file_path,
                 loss=loss, mask_loss=mask_loss, edge_loss=edge_loss,
                 mask_coef=mask_coef, edge_coef=edge_coef)
        print('results saved: ' + file_path)
    return


def predict_mask_v2(model_name, out_path=None):
    """input the images as a whole"""
    print('pre-trained model: ', model_name)
    test_preparer = SegPreparer(raw_im_path, train_stats_path)
    imgs = test_preparer.get_imgs()

    model = get_model(model_name)
    pred_masks = []
    pred_edges = []
    print('predicting...')
    for img in imgs:
        mask, edge = model.predict(img[np.newaxis, ...])      # output: (1, 772, 1060, 1)
        pred_masks.append(mask)
        pred_edges.append(edge)

    pred_masks = postprocess(pred_masks)
    # pred_edges = postprocess(pred_edges)

    img_names = get_filename(raw_im_path)
    if out_path is not None:
        for name, mask, edge in zip(img_names, pred_masks, pred_edges):
            save_img(mask, out_path + '/predMask_' + name[0:-21] + '.png')
            # save_img(edge, out_path + '/predEdge_' + img_name[0:-21] + '.png')
    else:
        for name, mask in zip(img_names, pred_masks):
            plt.figure()
            plt.imshow(mask, 'gray')
            plt.title(name)
            plt.show()
    return


# def predict_mask_(model_name):
#     """input the images in crops"""
#     print('pre-trained model: ', model_name)
#     test_preparer = SegPreparer(raw_im_path, train_stats_path)
#     imgs = test_preparer.get_crops()
#
#     model = tf.keras.models.load_model('./results/model/' + model_name,
#                                        custom_objects={'dice_loss': dice_loss,
#                                                        'dice_coef': dice_coef},
#                                        compile=True)
#     pred_mask, pred_edge = model.predict(imgs)
#
#     test_preparer.toImages(pred_mask, './results/predict/pred_mask_C3.png')
#     test_preparer.toImages(pred_edge, './results/predict/pred_edge_C3.png')
#
#     return


def postprocess(imgs):
    """post-process predicted masks/edges; imgs: a list of arrays"""
    imgs = np.concatenate(imgs, axis=0)
    assert np.max(imgs) <= 1, 'input img is not gray-scale image'

    imgs *= 255
    # imgs[imgs < THRESHOLD] = 0      # threshold

    MARGIN = 30  # because of Cropping2D layer
    imgs = np.pad(imgs[:, :, :, 0], ((0,), (MARGIN,), (MARGIN,)), 'symmetric').astype('uint8')
    imgs = np.pad(imgs, ((0, 0), (0, 0), (0, 8)), 'symmetric')  # because of new input shape
    return imgs


def save_img(img, file_path):
    cv2.imwrite(file_path, img)
    print('img saved: ', file_path)
    return


def overlay_img_mask(imgs_path, pred_masks_path, out_path=None):
    """overlay raw gray-scale image with predicted mask"""
    preparer = SegPreparer(imgs_path, None, mask_path=pred_masks_path)
    preparer.load_test_set()                # get reading paths
    img_names = get_filename(imgs_path)     # get images names

    for i_path, m_path, img_name in zip(preparer.img_list, preparer.mask_list, img_names):
        img = cv2.imread(i_path, 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        mask = cv2.imread(m_path, 0)
        regions = mask > 0

        channel_multiplier = [2, 0, 0]
        img = img.astype('float32')
        img[regions, :] *= channel_multiplier

        if out_path is not None:
            save_img(img.astype('uint8'), out_path + '/rawAndMask_' + img_name[0:-21] + '.png')
        else:
            plt.figure()
            plt.imshow(img.astype('uint8'))
            plt.show()

    return


def overlay_img_gt_mask(imgs_path, gt_masks_path, pred_masks_path, out_path):
    """overlay raw image with edges of the ground truth mask and the predicted mask"""
    preparer = SegPreparer(imgs_path, None, mask_path=pred_masks_path)
    preparer.load_test_set()
    pred_mask_list = preparer.mask_list     # get pred_mask paths

    preparer._mask_path = gt_masks_path
    preparer.load_test_set()                # get raw image and gt_mask paths

    img_names = get_filename(imgs_path)     # get raw image names

    for raw_path, gt_m_path, pred_m_path, img_name in zip(preparer.img_list, preparer.mask_list, pred_mask_list,
                                                          img_names):
        raw_img = cv2.imread(raw_path, 0)
        gt_mask = cv2.imread(gt_m_path, 0)
        pred_mask = cv2.imread(pred_m_path, 0)

        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)

        channel_addend0 = [0, 0, 250]   # gt            # [b, g, r] for cv2.imwrite()
        channel_addend1 = [0, 250, 0]   # pred
        raw_img = overlay_edg(raw_img, gt_mask, channel_addend0)
        raw_img = overlay_edg(raw_img, pred_mask, channel_addend1)
        if out_path is not None:
            cv2.putText(raw_img, "Red: Ground Truth",
                        (10, 810), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 250], thickness=2)
            cv2.putText(raw_img, " Green: Predicted Mask",
                        (350, 810), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 250, 0], thickness=2)
            save_img(raw_img.astype('uint8'), out_path + '/rawAndEdge_' + img_name[0:-21] + '.png')
        else:
            plt.figure()
            plt.imshow(raw_img.astype('uint8'))
            plt.title('Red: GroundTruth, Green: PredictedMask')

            plt.show()

    return


def overlay_edg(background, inp, addend):
    if background.dtype != 'float32':
        background = background.astype('float32')

    inp_edge = get_edge(inp, iter=1)
    edge_region = inp_edge > 0

    background[edge_region] *= [0, 0, 0]        # clean it before overwriting it
    background[edge_region] += addend
    return background


if __name__ == '__main__':
    # initialization
    model_name = 'vUnet_FAK_N1_08.hdf5'
    raw_im_path = 'DataSet_label/Human_Muscle_PF573228/sample_test'
    # gt_mask_path = 'DataSet_label/FAK_N4/test_mask'
    # pred_mask_path = 'results/predict/FAK_N4/N4_model_01/predMask'
    # rawAndEdge_path = 'results/predict/FAK_N4/N4_model_01/rawAndEdges'
    train_stats_path = 'DataSet_label/FAK_N4/train/train_mean_std.npz'
    batch_size = 16

    # # get the loss and coef on Test set
    # evaluate_model(model_name, out_path=pred_mask_path)

    # # overlay raw image with edges of predicted mask and gt_mask
    predict_mask_v2(model_name, out_path=None)
    # overlay_img_gt_mask(raw_im_path, gt_mask_path, pred_mask_path, out_path=rawAndEdge_path)

    # # overlay raw image with predicted mask
    # rawAndMask_path = 'results/predict/FAK_N3/N1_model_08/rawAndMask'
    # overlay_img_mask(raw_im_path, pred_mask_path)

