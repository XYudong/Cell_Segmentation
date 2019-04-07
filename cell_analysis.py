from analysis_utils import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from os import path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def save_fea_vector(path_list):
    for mask_path in path_list:
        img_list = sorted(glob.glob(path.join(mask_path, '*.png')))
        print(str(len(img_list)) + ' images')
        extractor = FeatureExtractor()
        vectors = []
        for img_path in img_list:
            mask = cv2.imread(img_path, 0)
            fea_vec = extractor.get_feature_vec(mask)
            vectors.append(fea_vec)

        vectors = np.stack(vectors, axis=0)
        # column-wise normalization
        for j in range(vectors.shape[1]):
            v = vectors[:, j]
            vectors[:, j] = (v - np.mean(v)) / (np.std(v) + 1e-6)

        print('shape: ', vectors.shape)  # (N, F)
        np.save(mask_path + '/fea_vectors_00.npy', vectors)
        print('features saved: ', mask_path)


def visualize_pca(path_list):
    # loading feature vectors
    fea_vectors = []
    idx = [0]
    for fea_path in path_list:
        v = np.load(fea_path + '/fea_vectors_00.npy')
        fea_vectors.append(v)
        idx.append(v.shape[0] + idx[-1])

    fea_vectors = np.concatenate(fea_vectors, axis=0)
    print(fea_vectors.shape)

    FAK_names = ['FAK_' + s for s in ['N1', 'N2', 'N3', 'N4']]
    DMSO_names = ['DMSO_' + s for s in ['N1', 'N2', 'N3', 'N4']]

    # initialize PCA
    pca = PCA(n_components=2)
    fea_low = pca.fit_transform(fea_vectors)

    plt.figure()
    for i, dataset in enumerate(FAK_names):
        plt.scatter(fea_low[idx[i]:idx[i + 1] - 1, 0], fea_low[idx[i]:idx[i + 1] - 1, 1], marker='o', label=dataset)

    for i, dataset in enumerate(DMSO_names):
        i += len(FAK_names)
        plt.scatter(fea_low[idx[i]:idx[i + 1] - 1, 0], fea_low[idx[i]:idx[i + 1] - 1, 1], marker='+', label=dataset)

    plt.title('PCA of Cell Features')
    plt.legend()

    plt.show()


def visualize_tsne(path_list):
    fea_vectors = []
    idx = [0]
    for fea_path in path_list:
        v = np.load(fea_path + '/fea_vectors_00.npy')
        fea_vectors.append(v)
        idx.append(v.shape[0] + idx[-1])

    fea_vectors = np.concatenate(fea_vectors, axis=0)
    print(fea_vectors.shape)

    FAK_names = ['FAK_' + s for s in ['N1', 'N2', 'N3', 'N4']]
    DMSO_names = ['DMSO_' + s for s in ['N1', 'N2', 'N3', 'N4']]

    for perplexity in [20, 25, 30, 35, 40]:
        tsne = TSNE(n_components=2, perplexity=perplexity,
                    learning_rate=20, n_iter=5000)
        fea_low = tsne.fit_transform(fea_vectors)

        plt.figure()
        for i, dataset in enumerate(FAK_names):
            plt.scatter(fea_low[idx[i]:idx[i+1] - 1, 0], fea_low[idx[i]:idx[i+1] - 1, 1], marker='o', label=dataset)

        for i, dataset in enumerate(DMSO_names):
            i += len(FAK_names)
            plt.scatter(fea_low[idx[i]:idx[i+1] - 1, 0], fea_low[idx[i]:idx[i+1] - 1, 1], marker='+', label=dataset)

        plt.title('t-SNE of Cell Features, perplexity: ' + str(perplexity))
        plt.legend()

    plt.show()


if __name__ == '__main__':
    pred_mask_list = ['results/predict/MM_FAK_N1/N4_model_04/predMask',
                      'results/predict/MM_FAK_N2/N4_model_04/predMask',
                      'results/predict/MM_FAK_N3/N4_model_04/predMask',
                      'results/predict/MM_FAK_N4/N4_model_04/predMask',
                      'results/predict/MM_DMSO_N1/N4_model_04/predMask',
                      'results/predict/MM_DMSO_N2/N4_model_04/predMask',
                      'results/predict/MM_DMSO_N3/N4_model_04/predMask',
                      'results/predict/MM_DMSO_N4/N4_model_04/predMask']

    # First step:
    # save_fea_vector(pred_mask_list)

    # Second step:
    visualize_pca(pred_mask_list)
    # visualize_tsne(pred_mask_list)
