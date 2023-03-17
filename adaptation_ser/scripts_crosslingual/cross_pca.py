import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os
from shutil import copy2

COLORS = ['c', 'r', 'b', 'g']
size = 2
area = size ** 2
domain_dict = {0: 'iem', 1: 'rec'}
base_path = '/home/ddy17/projects/adaptation_ser'


def draw_pca(features, labels, fold_path, c_dict=None):
    pca = PCA(n_components=2)
    embed = pca.fit_transform(features)
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
    plt.savefig(os.path.join(fold_path, 'cross_lingual_pca.png'))
    plt.clf()


def draw_fold(fold_path, is_sample=True):
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


def rename_png(f2, f3, f4):
    in_base_dir = '/Users/ddy/Desktop/npys'
    out_base_dir = '/Users/ddy/Desktop/npys/pngs'
    os.makedirs(out_base_dir, exist_ok=True)
    f_path = os.path.join(in_base_dir, f2, f3, f4)
    png_names = [f_name for f_name in os.listdir(f_path) if 'png' in f_name]
    for png_name in png_names:
        in_path = os.path.join(f_path, png_name)
        png_name2 = png_name.replace('_', '')
        out_name = '-'.join(('ch3', f2, f3, f4, png_name2))
        out_path = os.path.join(out_base_dir, out_name)
        copy2(in_path, out_path)


if __name__ == '__main__':
    fold1 = 'npys'
    folds2 = ['baseline', 'mmd']
    folds3 = ['iemocap2recola', 'recola2iemocap']
    folds4 = ['arousal', 'valance']
    for fold2 in folds2:
        for fold3 in folds3:
            for fold4 in folds4:
                rename_png(fold2, fold3, fold4)
                # fold = '/'.join((fold1, fold2, fold3, fold4))
                # _fold_path = os.path.join(base_path, fold1, fold2, fold3, fold4)

                # draw_fold(f_path)
    #
    # fold = 'npys/baseline/iemocap2recola/arousal'
    # f_path = os.path.join(base_path, fold)
    # draw_fold(f_path)
