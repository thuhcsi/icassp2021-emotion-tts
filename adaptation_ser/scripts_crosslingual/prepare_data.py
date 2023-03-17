import tensorflow as tf
import random
import os
import numpy as np


def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def load_data(hp):
    data_metas = []  # data_meta: [(uid, spectrogram, seq_len, arousal, valance)]
    with open(hp.npy_meta_path, 'r') as meta_f:
        origin_metas = [meta.strip().split('|') for meta in meta_f.readlines()]
        for origin_meta in origin_metas:
            uid = os.path.splitext(origin_meta[0])[0].encode('utf-8')
            npy_path = os.path.join(hp.npy_dir, origin_meta[0])
            spec = np.load(npy_path)
            seq_len = spec.shape[0]
            arousal = int(origin_meta[1])
            valance = int(origin_meta[2])
            data_metas.append((uid, spec, seq_len, arousal, valance))
    random.shuffle(data_metas)
    valid_data_metas = data_metas[:hp.valid_size]
    train_data_metas = data_metas[hp.valid_size:]
    return train_data_metas, valid_data_metas


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
    for v_meta in valid_data_metas:
        valid_metas.append((v_meta[0], (v_meta[1] - mu) / std, v_meta[2], v_meta[3], v_meta[4]))
    return train_metas, valid_metas


def save_list2tfr(data_metas, tfr_path):
    with tf.python_io.TFRecordWriter(tfr_path) as writer:
        for uid, spec, seq_len, arousal, valance in data_metas:
            print('uid:', uid)
            example = tf.train.Example(features=tf.train.Features(feature={
                'uid': _byte_feature(uid),
                'spec': _byte_feature(spec.astype(np.float32).tostring()),
                'seq_len': _int64_feature(seq_len),
                'arousal': _int64_feature(arousal),
                'valance': _int64_feature(valance)
            }))
            writer.write(example.SerializeToString())


def process(hp):
    train_meta, test_meta = load_data(hp)
    print('norm ...')
    train_meta, test_meta = norm(train_meta, test_meta)
    if hp.valid_size == 0:
        tfr_path = os.path.join(hp.tfr_dir, 'all.tfr')
        save_list2tfr(train_meta, tfr_path)
    else:
        print('train.tfr')
        train_path = os.path.join(hp.tfr_dir, 'train.tfr')
        save_list2tfr(train_meta, train_path)
        print('vali.tfr')
        valid_path = os.path.join(hp.tfr_dir, 'valid.tfr')
        save_list2tfr(test_meta, valid_path)


if __name__ == '__main__':
    pass
