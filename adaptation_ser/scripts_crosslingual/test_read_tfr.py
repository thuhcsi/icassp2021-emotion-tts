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


def get_dataset(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_single_example)
    dataset = dataset.repeat(2)
    dataset = dataset.shuffle(2)
    dataset = dataset.padded_batch(2, padded_shapes={'uid': (),
                                                    'spec': [None, None],
                                                    'seq_len': (),
                                                    'arousal': (),
                                                    'valance': ()})
    # dataset = dataset.batch(1)
    # dataset = dataset.padded_batch(2, padded_shapes={'data': [None, None],
    #                                                  'len': (),
    #                                                  'label': ()})
    # dataset = dataset.padded_batch(2, padded_shapes=[None, None])
    return dataset


def main():
    data_set = get_dataset('/Users/bytedance/Projects/data/cx_data/recola/tfr/train.tfr')
    iterator = data_set.make_one_shot_iterator()
    next_item = iterator.get_next()
    loop_num = 2
    with tf.Session() as sess:
        for i in range(loop_num):
            t = sess.run(next_item)
            # print(t[0])
            print(t)
            print('\n')



if __name__ == '__main__':
    main()
