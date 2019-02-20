from analysis_utils import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from os import path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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
    for j in range(vectors.shape[1]):
        v = vectors[:, j]
        vectors[:, j] = (v - np.mean(v)) / (np.std(v) + 1e-6)

    print('shape: ', vectors.shape)  # (N, F)
    np.save(mask_path + '/fea_vectors_00.npy', vectors)
    print('features saved: ', mask_path)


def visualize_pca(fea_list):
    # loading feature vectors
    fea_vectors = []
    idx = [0]
    for fea_path in fea_list:
        v = np.load(fea_path + '/fea_vectors_00.npy')
        fea_vectors.append(v)
        idx.append(v.shape[0] - 1 + idx[-1])

    fea_vectors = np.concatenate(fea_vectors, axis=0)
    print(fea_vectors.shape)

    FAK_names = ['FAK_' + s for s in ['N1', 'N2', 'N3', 'N4']]
    DMSO_names = ['DMSO_' + s for s in ['N1', 'N2', 'N3', 'N4']]

    # initialize PCA
    pca = PCA(n_components=2)
    fea_low = pca.fit_transform(fea_vectors)

    plt.figure()
    for i, dataset in enumerate(FAK_names):
        plt.scatter(fea_low[idx[i]:idx[i + 1], 0], fea_low[idx[i]:idx[i + 1], 1], marker='o', label=dataset)

    for i, dataset in enumerate(DMSO_names):
        i += len(FAK_names)
        plt.scatter(fea_low[idx[i]:idx[i + 1], 0], fea_low[idx[i]:idx[i + 1], 1], marker='+', label=dataset)

    plt.title('PCA of Cell Features')
    plt.legend()

    plt.show()


def visualize_tsne(fea_list):
    fea_vectors = []
    idx = [0]
    for fea_path in fea_list:
        v = np.load(fea_path + '/fea_vectors_00.npy')
        fea_vectors.append(v)
        idx.append(v.shape[0] - 1 + idx[-1])

    fea_vectors = np.concatenate(fea_vectors, axis=0)
    print(fea_vectors.shape)

    FAK_names = ['FAK_' + s for s in ['N1', 'N2', 'N3', 'N4']]
    DMSO_names = ['DMSO_' + s for s in ['N1', 'N2', 'N3', 'N4']]

    for perplexity in [10, 15, 20, 30, 50]:
        tsne = TSNE(n_components=2, perplexity=perplexity)
        fea_low = tsne.fit_transform(fea_vectors)

        plt.figure()
        for i, dataset in enumerate(FAK_names):
            plt.scatter(fea_low[idx[i]:idx[i+1], 0], fea_low[idx[i]:idx[i+1], 1], marker='o', label=dataset)

        for i, dataset in enumerate(DMSO_names):
            i += len(FAK_names)
            plt.scatter(fea_low[idx[i]:idx[i+1], 0], fea_low[idx[i]:idx[i+1], 1], marker='+', label=dataset)

        plt.title('t-SNE of Cell Features, perplexity: ' + str(perplexity))
        plt.legend()

    plt.show()


if __name__ == '__main__':
    pred_mask_path = 'DataSet_label/DMSO_N4/GFP'
    # pred_mask_path = 'results/predict/FAK_N1/N1_model_08/predMask'
    # pred_mask_path = 'DataSet_label/FAK_N1/GFP_MASK_PNG'
    # save_fea_vector(pred_mask_path)

    pred_mask_list = ['results/predict/FAK_N1/N1_model_08/predMask',
                      'results/predict/FAK_N2/N1_model_08/predMask',
                      'results/predict/FAK_N3/N1_model_08/predMask',
                      'results/predict/FAK_N4/N1_model_08/predMask',
                      'DataSet_label/DMSO_N1/GFP',
                      'DataSet_label/DMSO_N2/GFP',
                      'DataSet_label/DMSO_N3/GFP',
                      'DataSet_label/DMSO_N4/GFP']
    visualize_pca(pred_mask_list)



