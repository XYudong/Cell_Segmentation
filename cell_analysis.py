from analysis_utils import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from os import path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def save_fea_vector(path_list, out_path):
    vectors = []
    idx_FAK = [0]
    count = 0
    for single_trial_path in path_list:
        # for each trial folder
        imgs_list = sorted(glob.glob(path.join(single_trial_path, '*.png')))
        print(str(len(imgs_list)) + ' images in ' + single_trial_path)
        extractor = FeatureExtractor()

        for single_img_path in imgs_list:
            # for each mask
            mask = cv2.imread(single_img_path, 0)
            fea_vec = extractor.get_fea_vec_0(mask)         # ndarray
            vectors.append(fea_vec)

        if count < 3:       # number of FAK_trial folders
            idx_FAK.append(idx_FAK[-1] + len(imgs_list))
            count += 1

    vectors = np.stack(vectors, axis=0)
    # column-wise normalization on the whole data matrix
    for j in range(vectors.shape[1]):
        v = vectors[:, j]
        vectors[:, j] = (v - np.mean(v)) / (np.std(v) + 1e-6)

    # save FAK trial data separately
    FAK_names = ['FAK_' + s for s in ['N4', 'N5', 'N6']]
    for i, dataset in enumerate(FAK_names):
        filename = dataset + '_fea_vectors_00.npy'
        np.save(path.join(out_path, filename), vectors[idx_FAK[i]:idx_FAK[i+1], :])

    np.save(path.join(out_path, 'DMSO_fea_vectors_00.npy'), vectors[idx_FAK[-1]:, :])
    print('shape: ', vectors.shape)  # (Num, Fea)
    print('features saved')


def load_fea(path_list):
    # loading feature vectors
    fea_vectors = []
    idx_list = [0]
    for fea_path in path_list:
        v = np.load(fea_path)
        fea_vectors.append(v)
        idx_list.append(v.shape[0] + idx_list[-1])
        print('feature vectors shape: ', v.shape)

    fea_vectors = np.concatenate(fea_vectors, axis=0)
    return fea_vectors, idx_list


def visualize_pca(path_list):
    fea_vectors, idx = load_fea(path_list)

    FAK_names = ['FAK_' + s for s in ['N4', 'N5', 'N6']]
    DMSO_names = ['DMSO_' + s for s in ['N4, N6']]

    # initialize PCA
    pca = PCA(n_components=2)
    fea_low = pca.fit_transform(fea_vectors)
    # fea_low = fea_vectors

    plt.figure()
    for i, dataset in enumerate(FAK_names):
        plt.scatter(fea_low[idx[i]:idx[i + 1], 0], fea_low[idx[i]:idx[i + 1], 1], marker='o', label=dataset)

    for i, dataset in enumerate(DMSO_names):
        i += len(FAK_names)
        plt.scatter(fea_low[idx[i]:idx[i + 1], 0], fea_low[idx[i]:idx[i + 1], 1], marker='+', label=dataset)

    plt.title('PCA of Cell Features')
    plt.legend()

    plt.show()


def visualize_tsne(path_list):
    fea_vectors, idx = load_fea(path_list)

    FAK_names = ['FAK_' + s for s in ['N4', 'N5', 'N6']]
    DMSO_names = ['DMSO_' + s for s in ['N4, N6']]

    for perplexity in [20, 25, 30, 35, 15]:
        tsne = TSNE(n_components=2, perplexity=perplexity,
                    learning_rate=20, n_iter=5000)
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
    root = 'results/predict/HumanMuscle'
    # predicted masks
    mask_list = ['HM_FAK_N4/N4_model_03/predMask',
                 'HM_FAK_N5/N4_model_03/predMask',
                 'HM_FAK_N6/N4_model_03/predMask',
                 'HM_DMSO_N4/N4_model_04/predMask',
                 'HM_DMSO_N6/N4_model_04/predMask']
    pred_mask_list = [path.join(root, mask) for mask in mask_list]
    fea_out_path = root

    # feature matrices extracted from the masks
    fea_list = ['FAK_N4_fea_vectors_00.npy',
                'FAK_N5_fea_vectors_00.npy',
                'FAK_N6_fea_vectors_00.npy',
                'DMSO_fea_vectors_00.npy']
    fea_list = [path.join(fea_out_path, fea) for fea in fea_list]

    # First step:
    # save_fea_vector(pred_mask_list, fea_out_path)

    # Second step:
    # visualize_pca(fea_list)
    visualize_tsne(fea_list)
