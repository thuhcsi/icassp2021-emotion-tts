import tensorflow as tf
import os


def source_parse_single_example(example_proto):
    features = {'uid': tf.FixedLenFeature([], tf.string),
                'mel': tf.FixedLenFeature([], tf.string),
                'seq_len': tf.FixedLenFeature([], tf.int64),
                'emo_idx': tf.FixedLenFeature([], tf.int64),
                'weight': tf.FixedLenFeature([], tf.float32)}
    parsed = tf.parse_single_example(example_proto, features=features)
    seq_len = tf.cast(parsed['seq_len'], tf.int32)
    emo_idx = tf.cast(parsed['emo_idx'], tf.int32)
    weight = tf.cast(parsed['weight'], tf.float32)
    mel = tf.reshape(tf.decode_raw(parsed['mel'], tf.float32), [seq_len, -1])
    return {'uid': parsed['uid'], 'mel': mel, 'seq_len': seq_len, 'emo_idx': emo_idx, 'weight': weight}


class SourceDataSet(object):

    def __init__(self, tfr_dir, hp, is_repeat=False):
        tfr_files = os.listdir(tfr_dir)
        tfr_paths = [os.path.join(tfr_dir, tfr_file) for tfr_file in tfr_files]
        d_set = tf.data.TFRecordDataset(tfr_paths)
        d_set = d_set.map(source_parse_single_example)
        if is_repeat:
            d_set = d_set.repeat()
            d_set = d_set.shuffle(2000)
        d_set = d_set.padded_batch(hp.batch_size, padded_shapes={'uid': (),
                                                                 'mel': [None, hp.feat_dim],
                                                                 'seq_len': (),
                                                                 'emo_idx': (),
                                                                 'weight': ()})
        self.d_set = d_set

    def get_iter(self):
        batch_iter = self.d_set.make_initializable_iterator()
        return batch_iter


def target_parse_single_example(example_proto):
    features = {'uid': tf.FixedLenFeature([], tf.string),
                'mel': tf.FixedLenFeature([], tf.string),
                'seq_len': tf.FixedLenFeature([], tf.int64)}
    parsed = tf.parse_single_example(example_proto, features=features)
    seq_len = tf.cast(parsed['seq_len'], tf.int32)
    mel = tf.reshape(tf.decode_raw(parsed['mel'], tf.float32), [seq_len, -1])
    return {'uid': parsed['uid'], 'mel': mel, 'seq_len': seq_len}


class TargetDataSet(object):

    def __init__(self, tfr_dir, hp, is_repeat=False):
        tfr_files = os.listdir(tfr_dir)
        tfr_paths = [os.path.join(tfr_dir, tfr_file) for tfr_file in tfr_files]
        d_set = tf.data.TFRecordDataset(tfr_paths)
        d_set = d_set.map(target_parse_single_example)
        if is_repeat:
            d_set = d_set.repeat()
            d_set = d_set.shuffle(2000)
        d_set = d_set.padded_batch(hp.batch_size, padded_shapes={'uid': (),
                                                                 'mel': [None, hp.feat_dim],
                                                                 'seq_len': ()})
        self.d_set = d_set

    def get_iter(self):
        batch_iter = self.d_set.make_initializable_iterator()
        return batch_iter
