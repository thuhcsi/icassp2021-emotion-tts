import re
import os
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

from os import path
from functools import partial
from collections import Counter

from sygst_hparams import hp
from .utils import tf_util, audio
from .speech_dataset import SpeechDataset


"""
total: 10039

categories:
{'neu': 1708, 'xxx': 2507, 'fru': 1849, 'ang': 1103, 'sad': 1084,
 'hap': 595, 'exc': 1041, 'sur': 107, 'oth': 3, 'fea': 40, 'dis': 2}
"""


class Iemocap(SpeechDataset):

    def __init__(self, hp=None, base_dir='data', data_dirs='wavs', **kwargs):
        super().__init__(hp, 'iemocap', base_dir, data_dirs, **kwargs)

        self.emo_dim_pivot = 2.5  # Remap emotion dimension labels to binary
        self.aro_map = {'low': 0, 'hig': 1}
        self.val_map = {'neg': 0, 'pos': 1}
        self.emo_map = {'neu': 0, 'ang': 1, 'hap': 2, 'exc': 2, 'sad': 3}
        self.splits = {'train': 0.9, 'valid': 0.1}

        # Reg exps extract: start, end, uid, emo_cls, val, aro, dom, text
        self.info_pa = re.compile(r'(Ses\w*)\s*(\w*).*\[')  # uid, emo_cls
        self.num_pa = re.compile(r'(\d+\.\d+)')  # start, end, val, aro, dom
        self.text_pa = re.compile(r'(Ses.*\w)\s\[.*\]:\s+(.*)')  # script text

    def _process_raw(self):
        self._info_lines, self._wav_paths = [], [[], []]
        session_base = path.join('Session{}', 'dialog', 'EmoEvaluation')
        for i in range(1, 6):
            session_dir = path.join(self.raw_dir, session_base.format(i))
            print('  processing dir: ', session_dir)
            file_names = next(os.walk(session_dir))[2]  # 获取session_dir下的所有文件名
            for name in file_names:
                self._process_one_raw(path.join(session_dir, name))

        # Post settings
        comment = '#index|cls|aro|val|dom|wav_sec|uid|text\n'
        self._info_lines.insert(0, comment)
        process_func = partial(audio.resample, self.hp)
        self._data_paths = [(self._wav_paths, process_func, 'process')]

        # done

    def _process_one_raw(self, meta_file):
        with open(meta_file, 'r') as fr:
            lines = fr.readlines()
        text_file = meta_file.replace('EmoEvaluation', 'transcriptions')
        with open(text_file, 'r') as fr:
            text_dict = dict([self.text_pa.findall(x)[0]
                             for x in fr if self.text_pa.findall(x)])

        wav_prefix = meta_file.replace('dialog', 'sentences').replace(
            'EmoEvaluation', 'wav')[: -4]  # drop out '.txt'

        for line in lines[2:]:
            if 'Ses' not in line:
                continue
            uid, emo_cls = self.info_pa.search(line).groups()
            sta_time, end_time, val, aro, dom = list(map(
                float, self.num_pa.findall(line)
            ))
            wav_sec = end_time - sta_time
            text = re.sub(r'\s+', ' ', text_dict[uid])  # drop extra spaces
            info = '{:06d}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}'.format(
                self._index, emo_cls, aro, val, dom, wav_sec, uid, text
            )

            src_wav = path.join(wav_prefix, uid + '.wav')
            dst_wav = path.join(self.data_dirs[0], f'{self._index:06d}.wav')

            self._index += 1
            self._info_lines.append(info + '\n')
            self._wav_paths[0].append(src_wav)
            self._wav_paths[1].append(dst_wav)
        # done

    def _trans_meta_ser(self, lines):
        def _remap_label(emo_label, aro_label, val_label):
            emo_idx = -1
            if emo_label in self.emo_map:
                emo_idx = self.emo_map[emo_label]
                emo_cnter[emo_idx] += 1
            aro_label = 'low' if float(aro_label) <= self.emo_dim_pivot else 'hig'
            val_label = 'neg' if float(val_label) <= self.emo_dim_pivot else 'pos'
            aro_idx = self.aro_map[aro_label]
            val_idx = self.val_map[val_label]
            aro_cnter[aro_idx] += 1
            val_cnter[val_idx] += 1
            return emo_idx, aro_idx, val_idx

        def _class_weight(cnter):
            weights = 1 / np.array(list(cnter.values()))
            weights /= np.mean(weights)
            return dict(zip(cnter, weights))

        lines = [x.split('|') for x in lines]
        emo_cnter, aro_cnter, val_cnter = Counter(), Counter(), Counter()

        # index|emo_cls|val|aro|dom|wav_sec|uid|text
        for line in lines:
            line[1: 4] = _remap_label(*line[1: 4])
        emo_ws = _class_weight(emo_cnter)
        aro_ws = _class_weight(aro_cnter)
        val_ws = _class_weight(val_cnter)

        emo_ws[-1] = 0
        emo_lines, dim_lines = [], []  # For emo classes and emo dimensions.
        for line in lines:
            weights = [emo_ws[line[1]], aro_ws[line[2]], val_ws[line[3]]]
            res = [*line[: 4], *weights, *line[5:]]
            res = '{}|{}|{}|{}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}\n'.format(*res)
            dim_lines.append(res)
            if line[1] != -1:
                emo_lines.append(res)

        comment = f'#aro map {self.aro_map} -> aro num {dict(aro_cnter)}\n'
        comment += f'#val map {self.val_map} -> val num {dict(val_cnter)}\n'
        comment += f'#emo map {self.emo_map} -> emo num {dict(emo_cnter)}\n'
        comment += '#index|emo|aro|val|emo_w|aro_w|val_w|wav_sec|uid|text\n'
        dim_lines.insert(0, comment)
        emo_lines.insert(0, comment)
        return [('dim_meta.txt', dim_lines), ('emo_meta.txt', emo_lines)]

    def _make_tfrecord_ser(self, line, *audio_features):
        # index|idx|cls|val|aro|dom|wav_sec|uid|text
        line = line.split('|')
        names = ['mel']
        num_frames = audio_features[0].shape[-1]  # mel
        audio_features = dict(zip(names, audio_features))
        audio_features = {k: tf_util.bytes_feature(audio_features[k].tobytes())
                          for k in audio_features}
        features = {
            'index': tf_util.int64_feature(int(line[0])),
            'emo_idx': tf_util.int64_feature(int(line[1])),
            'aro_idx': tf_util.int64_feature(int(line[2])),
            'val_idx': tf_util.int64_feature(int(line[3])),
            'emo_weight': tf_util.float_feature(float(line[4])),
            'aro_weight': tf_util.float_feature(float(line[5])),
            'val_weight': tf_util.float_feature(float(line[6])),
            'uid': tf_util.bytes_feature(bytes(line[8], 'utf-8')),
            'seq_len': tf_util.int64_feature(num_frames),
        }
        features.update(audio_features)
        res = tf.train.Example(features=tf.train.Features(feature=features))
        return res.SerializeToString()


if __name__ == '__main__':
    dset = Iemocap(hp, process=8, thread=8)
    # dset.process_raw()
    # dset.extract_mels('wavs')
    # dset.extract_specs('wavs')
    dset.transform_meta(dset._trans_meta_ser, action='transfer meta ser',
                        comment='get idx label and class weight')
    dset.make_tfrecords(dset._make_tfrecord_ser, 'dim_meta.txt',
                        dst_name='dim_tfrs', feat_names=['80_mels'], chops=50)
    dset.make_tfrecords(dset._make_tfrecord_ser, 'emo_meta.txt',
                        dst_name='emo_tfrs', feat_names=['80_mels'], chops=50)
