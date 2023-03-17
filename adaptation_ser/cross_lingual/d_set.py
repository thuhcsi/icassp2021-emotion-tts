import tensorflow as tf


def parse_single_example(example_proto):
    features = {'uid': tf.FixedLenFeature([], tf.string),
                'spec': tf.FixedLenFeature([], tf.string),
                'seq_len': tf.FixedLenFeature([], tf.int64),
                'arousal': tf.FixedLenFeature([], tf.int64),
                'valance': tf.FixedLenFeature([], tf.int64)}
    parsed = tf.parse_single_example(example_proto, features=features)
    seq_len = tf.cast(parsed['seq_len'], tf.int32)
    arousal = tf.cast(parsed['arousal'], tf.int32)
    valance = tf.cast(parsed['valance'], tf.int32)
    spec = tf.reshape(tf.decode_raw(parsed['spec'], tf.float32), [seq_len, -1])
    return {'uid': parsed['uid'], 'spec': spec, 'seq_len': seq_len, 'arousal': arousal, 'valance': valance}


class DataSet(object):
    def __init__(self, is_repeat, tfr_path, hp):
        d_set = tf.data.TFRecordDataset(tfr_path)
        d_set = d_set.map(parse_single_example)
        if is_repeat:
            d_set = d_set.repeat()
        d_set = d_set.shuffle(5000)
        d_set = d_set.padded_batch(hp.batch_size, padded_shapes={'uid': (),
                                                                 'spec': [None, hp.feat_dim],
                                                                 'seq_len': (),
                                                                 'arousal': (),
                                                                 'valance': ()})
        self.d_set = d_set

    def get_iter(self):
        batch_iter = self.d_set.make_initializable_iterator()
        return batch_iter
