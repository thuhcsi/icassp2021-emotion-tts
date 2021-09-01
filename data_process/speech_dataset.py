import os
import re
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

from os import path
from tqdm import tqdm
from functools import partial


from .dataset import Dataset
from .utils import audio
from .utils.func_tools import split_data


class SpeechDataset(Dataset):

    def __init__(self, hp, name, base_dir, data_dirs, **kwargs):
        super().__init__(hp, name, base_dir, data_dirs, **kwargs)

        self._index = 1  # initialize samples' index with 1

    def process_raw(self):
        super().process_raw()  # call for check the dir whether exist

        # Call the specific raw processing func in subclasses
        print(f'{self.name}: processing raw ...')
        self._process_raw()
        assert hasattr(self, '_info_lines') and hasattr(self, '_data_paths')

        # Write the meta infos to meta.txt
        print('  writing meta.txt ...')
        self.write_meta(self.meta_file, self._info_lines)

        # Copying or resampling wavs and other datas
        for paths, func, mode in self._data_paths:
            self.para_executor(func, *paths, mode=mode, num=self._config[mode],
                               desc='  [COPYING DATA]')
        self.log('process raw', self.raw_dir, [self.data_dirs[0], self.meta_file])
        print(f'{self.name} raw processing is finished.')

        del self._info_lines, self._data_paths
        # done

    def process_feature(self, features):
        pass

    def resample_wavs(self, src_dir=None, sr=None, mode='thread', extra=''):
        """
        Resampling all wavs at src_dir, the result wavs will be saved at dir:
            processed/datas/newrate_wavs

        # Arguments
            src_dir: the dir of wavs will be resampled
            sr: new rate, if given then use it, whether use self.hp.sample_rate
            mode: 'thread' or 'process', used for parallel executations
            extra: a extra info str will be added at end of the saved dir name
        """
        self.hp.sample_rate = sr if sr else self.hp.sample_rate
        src_dir = src_dir if src_dir else path.join(self.data_dir, 'raw_wavs')
        dst_dir = f'{self.hp.sample_rate}_{extra}_wavs'
        dst_dir = path.join(self.data_dir, dst_dir.replace('__', '_'))

        names = self.make_names(src_dir, dst_dir)
        if type(names) == str:
            return print(names)
        self.para_executor(partial(audio.resample, self.hp), *names, mode=mode,
                           num=self._config[mode], desc='[RESAMPLE WAVS]')
        self.log('resample wavs', src_dir, dst_dir, f'{self.hp.sample_rate}')

        # done

    def extract_base(self, func, src_dir, dst_name, mode='thread',
                     action='extract feature', comment=None):
        """A helper function for various feature extration"""
        dst_name = dst_name.replace('__', '_')
        dst_dir = path.join(self.feature_dir, dst_name)
        desc = f'[EXTRACT TO {dst_name}]'

        names = self.make_names(src_dir, dst_dir)
        if type(names) == str:
            return print(names)

        kwargs = {'mode': mode, 'num': self._config[mode], 'desc': desc}
        self.para_executor(func, *names, **kwargs)
        self.log(action, src_dir, dst_dir, comment)

    def extract_mels(self, wav_dir, mode='process', extra=''):
        self.extract_base(
            func=FeatureDecorator(audio.melspectrogram, self.hp),
            src_dir=path.join(self.data_dir, wav_dir),
            dst_name=f'{self.hp.num_mels}_{extra}_mels',
            mode=mode, action='extract mels',
        )

    def extract_mfccs(self, wav_dir, mode='process', extra=''):
        self.extract_base(
            func=FeatureDecorator(audio.mfcc, self.hp),
            src_dir=path.join(self.data_dir, wav_dir),
            dst_name=f'{self.hp.num_mfcc}_{extra}_mfccs',
            mode=mode, action='extract mfccs',
        )

    def extract_specs(self, wav_dir, mode='process', extra=''):
        self.extract_base(
            func=FeatureDecorator(audio.spectrogram, self.hp),
            src_dir=path.join(self.data_dir, wav_dir),
            dst_name=f'{self.hp.num_spec}_{extra}_specs',
            mode=mode, action='extract linear specs',
        )

    def transform_meta(self, func, src_name=None,
                       action='transfer meta', comment=None, **kwargs):
        """
        Transform a source meta file to the destination meta file. It do some
        common operations for training metas, e.g., drop out length-unsuitable
        items, remap and normalize labels, and process the text transscripts.
        """

        src_meta = path.join(self.processed_dir, src_name) if src_name else None
        src_meta = src_meta if src_meta else self.meta_file

        with open(src_meta, 'r') as fr:
            src_lines = [x.strip() for x in fr if x.lstrip()[0] != '#']

        dst_metas = func(src_lines, **kwargs)
        for dst_meta in dst_metas:
            dst_name, dst_lines = dst_meta
            dst_file = path.join(self.processed_dir, dst_name)
            print(f'Transfer meta from {src_meta} -> {dst_file} ...')
            self.write_meta(dst_file, dst_lines)
            self.log(action, src_meta, dst_file, comment)

    def make_tfrecords(self, func, meta_name, feat_names, dst_name,
                       chops=10, splits=None, split_key=None,
                       action='make tfrecords', comment=None, **kwargs):
        """
        Make tf records using the meta file 'meta_name' and the feature dirs
        'feat_names', and the result tfr files will be saved at 'dst_name'

        # Arguments
            func: create and serialize a single example. The call signature is
                func(line, *feature_arrs, **kwargs), while the first arg 'line'
                is one line in meta file, feature_arrs are the np arrays loaded
                at feature dirs. So, you need to implement the this func with
                the above signature.
            chops: create chops tfr files instead of a single tfr file for
                better IO performance. It is set according by your dataset size.
                Please be sure each chopped file will be at leased >10MB.
            split_key: a function that take a meta line as input and return the
                sample class. It is used to split even for each class.
        """
        # Process file names and paths
        meta_file = path.join(self.processed_dir, meta_name)
        feat_names = [feat_names] if type(feat_names) == str else feat_names
        feat_dirs = [path.join(self.feature_dir, x) for x in feat_names]
        dst_dir = path.join(self.tf_dir, dst_name)
        # if path.isdir(dst_dir) and os.listdir(dst_dir):
        #     return print(f'The {dst_dir} is already existed, so exit with no thing.')
        os.makedirs(dst_dir, exist_ok=True)

        # Read the meta file
        print(f'Making TF Records using {meta_name} {feat_names} -> {dst_dir}')
        with open(meta_file, 'r') as fr:
            lines = [x.strip() for x in fr if x.lstrip()[0] != '#']

        def write_records(lines, name, chops):
            pa = re.compile(r'(\d+)\s*\|')
            sub_nums = [len(lines) // chops] * chops
            sub_nums[-1] += len(lines) % chops
            for i in range(chops):
                tfr_file = path.join(dst_dir, f'{name}_{i:03d}.tfr')
                with tf.io.TFRecordWriter(tfr_file) as fw:
                    for j in tqdm(range(sub_nums[i]), desc=f'[Writing {i}-th file]'):
                        line = lines[i * (len(lines) // chops) + j]
                        feat_name = pa.match(line).group(1) + '.npy'  # index + '.npy'
                        feats = [path.join(d, feat_name) for d in feat_dirs]
                        feats = [np.load(f).astype(np.float32) for f in feats]
                        record = func(line, *feats, **kwargs)
                        fw.write(record)

        # Perform split and write tf record files
        splits = splits or (self.hp.splits if hasattr(self.hp, 'splits') else None)
        if not splits:  # Do not perform split
            write_records(lines, 'data', chops)
        else:
            assert isinstance(splits, dict)
            total = len(lines)
            split_names = list(splits.keys())
            split_lines = split_data(lines, splits.values(), split_key)
            chops = [round(chops * len(s) / total) for s in split_lines]
            for name, lines, chop in zip(split_names, split_lines, chops):
                write_records(lines, name, chop)
        # Write logs
        self.log(action, [meta_file] + [feat_dirs], dst_dir, comment)

    def make_names(self, data_dir, feat_dir):
        if path.isdir(feat_dir) and os.listdir(feat_dir):
            return f'The {feat_dir} is already existed, so exit with no thing.'
        datas = os.listdir(data_dir)
        data_names = [path.join(data_dir, d) for d in datas]
        feat_names = [path.join(feat_dir, path.splitext(d)[0]) for d in datas]
        os.makedirs(feat_dir, exist_ok=True)
        return data_names, feat_names


class FeatureDecorator:
    """
    This decorator is used wrap the feature extraction functions in audio
    module, such as audio.melspectrogram(), audio.spectrogram(). These
    functions originally return a np.array object. Now we wrapper them to
    save the feature array to disk and no returns. You now pass the src
    file and dst file, it will load data from src and save feature to dst.
    we also partial the hp arg.

    Note: when used for parallel exectuation, we can not use a nested
    wrapper function. so, we wrapper it using this class.
    """
    def __init__(self, func, hp, *args, **kwargs):
        self.func = func
        self.hp = hp
        self.args = args
        self.kwargs = kwargs

    def __call__(self, src, dst):
        feat = self.func(self.hp, src, *self.args, **self.kwargs)
        np.save(dst, feat.astype(np.float32))   # librosa returns float64 ?
