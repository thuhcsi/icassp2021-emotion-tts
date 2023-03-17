import tensorflow as tf
import numpy as np


def byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_mean_std(train_data_metas):
    frame_nums = 0
    freq_wise_sum = 0
    for meta in train_data_metas:
        meta_sum = np.sum(meta[1], axis=0)
        freq_wise_sum += meta_sum
        frame_nums += meta[2]
    mu = freq_wise_sum / frame_nums
    fwise_var_sum = 0
    for meta in train_data_metas:
        var_sum = np.sum((meta[1] - mu) ** 2, axis=0)
        fwise_var_sum += var_sum
    var = fwise_var_sum / frame_nums
    std = np.sqrt(var)
    return mu, std


def norm(train_data_metas, valid_data_metas):
    mu, std = get_mean_std(train_data_metas)
    train_metas = list()
    valid_metas = list()
    for t_meta in train_data_metas:
        train_metas.append((t_meta[0], (t_meta[1] - mu) / std, t_meta[2], t_meta[3], t_meta[4]))
    if valid_data_metas is not None:
        for v_meta in valid_data_metas:
            valid_metas.append((v_meta[0], (v_meta[1] - mu) / std, v_meta[2], v_meta[3], v_meta[4]))
    return train_metas, valid_metas


def norm_tgt(train_data_metas, valid_data_metas):
    mu, std = get_mean_std(train_data_metas)
    train_metas = list()
    valid_metas = list()
    for t_meta in train_data_metas:
        train_metas.append((t_meta[0], (t_meta[1] - mu) / std, t_meta[2]))
    if valid_data_metas is not None:
        for v_meta in valid_data_metas:
            valid_metas.append((v_meta[0], (v_meta[1] - mu) / std, v_meta[2]))
    return train_metas, valid_metas
