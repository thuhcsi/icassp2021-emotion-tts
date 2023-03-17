import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os

COLORS = ['c', 'r', 'b', 'g']
size = 2
area = size ** 2

base_path = '/home/ddy17/projects/adaptation_ser'


def draw_pca(features, labels, fold_path, c_dict=None):
    # pca = PCA(n_components=2)
    # embed = pca.fit_transform(features)
    embed = TSNE(n_components=2).fit_transform(features)
    i = 0
    for c in set(list(labels)):
        if c_dict:
            label = c_dict[c]
        else:
            label = str(c)
        c_embed = embed[labels == c]
        if i < len(COLORS):
            plt.scatter(c_embed[:, 0], c_embed[:, 1], label=label, color=COLORS[i], s=area)
        else:
            plt.scatter(c_embed[:, 0], c_embed[:, 1], label=label, s=area)
        i += 1
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(fold_path, 'cross_lingual_tsne_full.png'))
    plt.clf()


def draw_fold(fold_path, is_sample=True):
    domain_dict = {0: 'iem', 1: 'rec'}
    iemocap_np = np.load(os.path.join(fold_path, 'iemocap.npy'))
    recola_np = np.load(os.path.join(fold_path, 'recola.npy'))
    if is_sample:
        np.random.shuffle(iemocap_np)
        np.random.shuffle(recola_np)
        iemocap_np = iemocap_np[::3]
        recola_np = recola_np[::3]
    iemocap_label = np.zeros((len(iemocap_np)), dtype=np.int32)
    recola_label = np.ones((len(recola_np)), dtype=np.int32)
    features = np.concatenate((iemocap_np, recola_np), axis=0)
    labels = np.concatenate((iemocap_label, recola_label))
    draw_pca(features, labels, fold_path, domain_dict)


if __name__ == '__main__':
    fold1 = 'npys'
    folds2 = ['baseline', 'mmd']
    folds3 = ['iemocap2recola', 'recola2iemocap']
    folds4 = ['arousal', 'valance']
    for fold2 in folds2:
        for fold3 in folds3:
            for fold4 in folds4:
                fold = '/'.join((fold1, fold2, fold3, fold4))
                print(fold)
                f_path = os.path.join(base_path, fold1, fold2, fold3, fold4)
                draw_fold(f_path, False)
