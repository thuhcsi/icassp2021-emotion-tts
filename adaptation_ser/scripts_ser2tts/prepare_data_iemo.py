import tensorflow as tf
import os
import numpy as np
import random
import math
import argparse
import prepare_data_util as pd_util

impro_emo_weight = [0.51, 1.96, 0.60, 0.93]
emo_weight = [0.77, 1.20, 0.81, 1.22]

arousal_weight = [1.378, 0.622]
valance_weight = [0.933, 1.067]


# npy_dir
# meta_path
# tfr_dir
# valid_size


def load_data(npy_dir, meta_path, valid_size, is_impro_only=False, label_type=None):
    print('in load data')
    print('label type:', label_type)
    meta_list = []
    if label_type is None:
        if is_impro_only:
            weight_list = impro_emo_weight
        else:
            weight_list = emo_weight
    elif label_type == 'arousal':
        print('label_type_arousal')
        weight_list = arousal_weight
    elif label_type == 'valance':
        print('label_type valance')
        # print('label type:', label_type)
        weight_list = valance_weight
    else:
        return Exception('Unseen label type')
    with open(meta_path, 'r') as meta_f:
        for line in meta_f:
            eles = line.strip().split('|')
            if len(eles) > 1:
                meta_list.append(eles)
    random.shuffle(meta_list)
    data_metas = []
    for meta_eles in meta_list:
        uid = meta_eles[0].encode('utf-8')
        npy_path = os.path.join(npy_dir, meta_eles[0] + '.npy')
        mel = np.load(npy_path)
        seq_len = mel.shape[0]
        emo_idx = int(meta_eles[2])
        weight = weight_list[emo_idx]
        data_metas.append((uid, mel, seq_len, emo_idx, weight))
    valid_data_metas = data_metas[:valid_size]
    train_data_metas = data_metas[valid_size:]
    return train_data_metas, valid_data_metas


def savelist2tfr(data_metas, tfr_dir, tf_name):
    max_examples = 15000
    sub_num = math.ceil(len(data_metas) / max_examples)
    for i in range(sub_num):
        tfr_path = os.path.join(tfr_dir, tf_name + '_' + str(i) + '.tfr')
        end_idx = (i + 1) * max_examples if (i + 1) * max_examples < len(data_metas) else len(data_metas)
        sub_data_metas = data_metas[i * max_examples:end_idx]
        with tf.python_io.TFRecordWriter(tfr_path) as writer:
            for uid, mel, seq_len, emo_idx, weight in sub_data_metas:
                print('uid:', uid)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'uid': pd_util.byte_feature(uid),
                    'mel': pd_util.byte_feature(mel.astype(np.float32).tostring()),
                    'seq_len': pd_util.int64_feature(seq_len),
                    'emo_idx': pd_util.int64_feature(emo_idx),
                    'weight': pd_util.float_feature(np.float32(weight))}))
                writer.write(example.SerializeToString())


def process(npy_dir, meta_path, valid_size, tfr_dir, is_impro_only=False, label_type=None):
    train_meta, test_meta = load_data(npy_dir, meta_path, valid_size, is_impro_only, label_type=label_type)
    print('norm ...')
    train_meta, test_meta = pd_util.norm(train_meta, test_meta)
    if valid_size == 0:
        tfr_name = 'all'
        savelist2tfr(train_meta, tfr_dir, tfr_name)
    else:
        print('save train tfr ...')
        train_name = 'train'
        savelist2tfr(train_meta, tfr_dir, train_name)
        print('save valid tfr ...')
        valid_name = 'valid'
        savelist2tfr(test_meta, tfr_dir, valid_name)


def arg_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_dir',
                        default='/home/ddy17/ser_data/ser2tts/iemo/npys_v2')
    parser.add_argument('--meta_path',
                        default='/home/ddy17/ser_data/ser2tts/iemo/meta_valance')
    parser.add_argument('--valid_size',
                        default=900)
    parser.add_argument('--tfr_dir',
                        default='/home/ddy17/ser_data/ser2tts/iemo/tfr_valance')
    parser.add_argument('--label_type',
                        default='valance')
    args = parser.parse_args()
    process(args.npy_dir, args.meta_path, args.valid_size, args.tfr_dir, label_type=args.label_type)


if __name__ == '__main__':
    arg_main()
