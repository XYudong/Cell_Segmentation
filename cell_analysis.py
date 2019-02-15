from analysis_utils import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from os import path
from sklearn import manifold
from sklearn.decomposition import PCA


def save_fea_vector(mask_path):
    img_list = sorted(glob.glob(path.join(mask_path, '*.png')))
    print(str(len(img_list)) + ' images')
    extractor = FeatureExtractor()
    vectors = []
    for img_path in img_list:
        mask = cv2.imread(img_path, 0)
        fea_vec = extractor.get_feature_vec(mask)
        vectors.append(fea_vec)

    vectors = np.stack(vectors, axis=0)
    print('shape: ', vectors.shape)
    np.save(mask_path + '/fea_vectors.npy', vectors)
    print('features saved: ', mask_path)


def visualize_fea(fea_list):
    pca = PCA(n_components=2)
    plt.figure()
    names = ['N1', 'N2', 'N3', 'N4']
    for fea_path, dataset in zip(fea_list, names):
        fea_vectors = np.load(fea_path + '/fea_vectors.npy')
        print(fea_vectors.shape)
        fea_low = pca.fit_transform(fea_vectors)
        var_ratio = pca.explained_variance_ratio_
        print(var_ratio)

        plt.scatter(fea_low[:, 0], fea_low[:, 1], marker='o', label=dataset)

    plt.title('PCA of Cell Features')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    pred_mask_path = 'results/predict/FAK_N2/N1_model_08/predMask'
    # save_fea_vector(pred_mask_path)

    pred_mask_list = ['results/predict/FAK_N1/N1_model_08/predMask',
                      'results/predict/FAK_N2/N1_model_08/predMask',
                      'results/predict/FAK_N3/N1_model_08/predMask',
                      'results/predict/FAK_N4/N1_model_08/predMask']
    visualize_fea(pred_mask_list)



