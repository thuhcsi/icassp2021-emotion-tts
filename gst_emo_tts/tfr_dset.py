import os
import glob
import tensorflow as tf

# from taco2_hparams import hp
# from hparams import hparams as hp
# cpu_num = os.cpu_count()

_pad = 0.
_pad_emo = 0.25
_pad_bemo = 0.5
_pad_token = 1.  # stop token pad 1. for marking sequences finished


def parse_single_example(example_proto):
    features = {'uid': tf.FixedLenFeature([], tf.string),
                'inputs': tf.FixedLenFeature([], tf.string),
                'input_lengths': tf.FixedLenFeature([], tf.int64),
                'mel_targets': tf.FixedLenFeature([], tf.string),
                'linear_targets': tf.FixedLenFeature([], tf.string),
                'spec_lengths': tf.FixedLenFeature([], tf.int64),
                'soft_emo_labels': tf.FixedLenFeature([], tf.string),
                'soft_arousal_labels': tf.FixedLenFeature([], tf.string),
                'soft_valance_labels': tf.FixedLenFeature([], tf.string),
                'arousal_embedding': tf.FixedLenFeature([], tf.string),
                'valance_embedding': tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(example_proto, features=features)
    inputs = tf.decode_raw(parsed['inputs'], tf.int32)
    input_lengths = tf.cast(parsed['input_lengths'], tf.int32)
    spec_lengths = tf.cast(parsed['spec_lengths'], tf.int32)
    mel_targets = tf.reshape(tf.decode_raw(parsed['mel_targets'], tf.float32), [spec_lengths, -1])
    linear_targets = tf.reshape(tf.decode_raw(parsed['linear_targets'], tf.float32), [spec_lengths, -1])
    soft_emo_labels = tf.decode_raw(parsed['soft_emo_labels'], tf.float32)
    soft_arousal_labels = tf.decode_raw(parsed['soft_arousal_labels'], tf.float32)
    soft_valance_labels = tf.decode_raw(parsed['soft_valance_labels'], tf.float32)
    arousal_embedding = tf.decode_raw(parsed['arousal_embedding'], tf.float32)
    valance_embedding = tf.decode_raw(parsed['valance_embedding'], tf.float32)
    return {'uid': parsed['uid'],
            'inputs': inputs,
            'input_lengths': input_lengths,
            'mel_targets': mel_targets,
            'linear_targets': linear_targets,
            'spec_lengths': spec_lengths,
            'soft_emo_labels': soft_emo_labels,
            'soft_arousal_labels': soft_arousal_labels,
            'soft_valance_labels': soft_valance_labels,
            'arousal_embedding': arousal_embedding,
            'valance_embedding': valance_embedding}


def parse_single_example_for_merge_emo_feature(example_proto):
    features = {'uid': tf.FixedLenFeature([], tf.string),
                'inputs': tf.FixedLenFeature([], tf.string),
                'input_lengths': tf.FixedLenFeature([], tf.int64),
                'mel_targets': tf.FixedLenFeature([], tf.string),
                'linear_targets': tf.FixedLenFeature([], tf.string),
                'spec_lengths': tf.FixedLenFeature([], tf.int64),
                'soft_emo_labels': tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(example_proto, features=features)
    return parsed


def load_for_merge_emo_features(tfr_dir):
    file_pattern = os.path.join(tfr_dir, '*.tfr')
    tfrecord_files_num = len(glob.glob(file_pattern))
    tfrecord_files = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    dataset = tfrecord_files.apply(
        tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset,
                                                 cycle_length=min(
                                                     tfrecord_files_num,
                                                     128),
                                                 block_length=1))
    dataset = dataset.map(parse_single_example_for_merge_emo_feature,
                          num_parallel_calls=os.cpu_count())
    return dataset


def load_for_prepare_meta_from_tfr(tfr_dir):
    file_pattern = os.path.join(tfr_dir, '*.tfr')
    tfrecord_files_num = len(glob.glob(file_pattern))
    tfrecord_files = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    dataset = tfrecord_files.apply(
        tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset,
                                                 cycle_length=min(
                                                     tfrecord_files_num,
                                                     128),
                                                 block_length=1))
    dataset = dataset.map(parse_single_example,
                          num_parallel_calls=os.cpu_count())
    return dataset


class TFDataSet(object):
    def __init__(self,
                 hp,
                 tfr_dir,
                 cache_path=None,
                 valid_batches=None,
                 load_for_rayhame=False):
        """Load bc2013 dataset as a tf.data.Dataset object for training tts models
        # Arguments
            tfr_dir: the path of tf record files for bc2013
            batch_size: the batch size used for training and evaluating
            valid_batches: split the first 'valid_bathes' batch data as validation
                set, if None, will return the whole dataset as training set
            outputs_per_step: emit this number of frames at each tacotron
                decoder time step, it is passed to trim the spectrum lengths
            load_for_rayhame: whether load  data for training Rayhame's Tacotron2
                model, if True, wu will trim down the spectrums and compute the
                stop token targets
        """

        self.tfr_dir = tfr_dir
        self.cache_path = cache_path
        self.num_mels = hp.num_mels
        self.num_spec = hp.num_spec
        self.batch_size = hp.batch_size
        self.outputs_per_step = hp.outputs_per_step
        self.valid_batches = valid_batches
        self.load_for_rayhame = load_for_rayhame

    def load(self):

        # Load the tf record files to a tf.data.Dataset object
        auto_tune = tf.data.experimental.AUTOTUNE
        file_pattern = os.path.join(self.tfr_dir, '*.tfr')
        tfrecord_files_num = len(glob.glob(file_pattern))
        tfrecord_files = tf.data.Dataset.list_files(file_pattern, shuffle=True)
        dataset = tfrecord_files.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                cycle_length=min(tfrecord_files_num, 128),
                block_length=1
            )
        )

        # Deserialize each tf record example to a dict of Tensors
        dataset = dataset.map(lambda x: parse_single_example(x),
                              num_parallel_calls=auto_tune)

        # Filter sampels by spectrum lengths for removing some mismatched text-audio pairs
        def len_filter(x):
            return tf.logical_not(
                tf.logical_or(
                    tf.logical_or(x['spec_lengths'] < 80, x['spec_lengths'] > 800),
                    tf.logical_and(x['input_lengths'] < 70, x['spec_lengths'] > 700)
                )
            )
        dataset = dataset.filter(len_filter)

        def trim_down_lengths(x, prepare_stop_targets=True):
            r = self.outputs_per_step
            spec_len = x['spec_lengths']
            trim_len = tf.cast(spec_len / r, dtype=tf.int32) * r
            x['mel_targets'] = x['mel_targets'][: trim_len]
            x['linear_targets'] = x['linear_targets'][: trim_len]
            x['spec_lengths'] = trim_len
            if prepare_stop_targets:
                x['token_targets'] = tf.concat([tf.zeros(trim_len - r), tf.ones(r)], axis=0)
            return x
        # Load for rayhame' tacotron2 model
        if self.load_for_rayhame:
            assert self.valid_batches is not None
            assert self.outputs_per_step is not None
            dataset = dataset.map(trim_down_lengths, auto_tune)

        # Maybe split the valid dataset and training dataset
        valid_dataset = None
        if self.valid_batches:
            valid_size = self.valid_batches * self.batch_size
            dataset = dataset.shuffle(buffer_size=10000)
            valid_dataset = dataset.take(valid_size)  # validation set
            dataset = dataset.skip(valid_size)        # training set

        # Perform a bucket and padded batch transform for training set
        bucket_num = 10
        bucket_batch_sizes = [self.batch_size] * bucket_num
        bucket_boundaries = [25, 40, 55, 70, 85, 100, 135, 170, 220]
        padded_shapes = {'uid': [],
                         'inputs': [None],
                         'input_lengths': [],
                         'mel_targets': [None, self.num_mels],
                         'linear_targets': [None, self.num_spec],
                         'spec_lengths': [],
                         'soft_emo_labels': [None],
                         'soft_arousal_labels': [None],
                         'soft_valance_labels': [None],
                         'arousal_embedding': [256],
                         'valance_embedding': [256]}
        padded_values = {'uid': '\0',
                         'inputs': 0,
                         'input_lengths': 0,
                         'mel_targets': _pad,
                         'linear_targets': _pad,
                         'spec_lengths': 0,
                         'soft_emo_labels': _pad_emo,
                         'soft_arousal_labels': _pad_bemo,
                         'soft_valance_labels': _pad_bemo,
                         'arousal_embedding': _pad,
                         'valance_embedding': _pad}

        if self.load_for_rayhame:
            padded_shapes.update({'token_targets': [None]})
            padded_values.update({'token_targets': _pad_token})

        dataset = dataset.apply(
            tf.data.experimental.bucket_by_sequence_length(
                lambda x: x['input_lengths'],
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes,
                padded_shapes=padded_shapes,
                padding_values=padded_values,
                pad_to_bucket_boundary=False,
                no_padding=False
            )
        )

        # Shffle and repeat infinitely and prefetch 10 batches
        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=128))
        dataset = dataset.prefetch(buffer_size=10)   # Prefetch 10 batch of samples
        if self.cache_path:  # not None an not ''
            # assert os.path.isdir(self.cache_path)
            # dataset = dataset.cache(os.path.join(self.cache_path, 'cached_bc2013'))
            pass   # cache 有问题, 直接爆磁盘和内存

        # Perform padded batch transform for validation dataset
        if valid_dataset is not None:
            valid_dataset = valid_dataset.apply(
                tf.data.experimental.shuffle_and_repeat(self.valid_batches, count=1))
            valid_dataset = valid_dataset.padded_batch(
                self.batch_size, padded_shapes, padded_values)
            valid_dataset = valid_dataset.cache()

        self.dataset = dataset
        self.valid_dataset = valid_dataset

    def get_train_next(self):
        if not hasattr(self, 'dataset'):
            self.load()
        train_next = self.dataset.make_one_shot_iterator().get_next()
        return train_next

    def get_valid_iter_and_next(self):
        assert self.valid_batches is not None
        if not hasattr(self, 'valid_dataset'):
            self.load()
        init_iter = self.valid_dataset.make_initializable_iterator()
        return init_iter.initializer, init_iter.get_next()


def test():
    pass


if __name__ == '__main__':
    test()
