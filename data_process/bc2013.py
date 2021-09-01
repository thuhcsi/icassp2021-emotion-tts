import os
import numpy as np
import tensorflow.compat.v1 as tf

from os import path
from tqdm import tqdm
from functools import partial
from collections import Counter

from sygst_hparams import hp
from .utils import tf_util, audio
from .speech_dataset import SpeechDataset


class BC2013(SpeechDataset):

    def __init__(self, hp=None, base_dir='data', data_dirs='wavs', **kwargs):
        super().__init__(hp, 'bc2013', base_dir, data_dirs, **kwargs)

    def _process_raw(self):
        self._info_lines, self._wav_paths = [], [[], []]
        wav_base = path.join(self.raw_dir, 'wavs')
        sub_dirs = os.listdir(wav_base)

        for sub_dir in tqdm(sub_dirs, desc='  [PROCESS META]'):
            wav_dir = path.join(wav_base, sub_dir)
            wav_files = [path.join(wav_dir, x) for x in os.listdir(wav_dir)]
            for wav_file in wav_files:
                txt_file = wav_file.replace('wav', 'txt')
                # print(txt_file, wav_file)
                # Process meta txt files
                with open(txt_file, 'r') as fr:
                    lines = [line.strip() for line in fr.readlines()]
                    assert len(lines) == 1, f'read {txt_file} error: {lines}'
                uid = path.join(sub_dir, path.split(txt_file)[-1])[: -4]
                info_line = f'{self._index:06d}|{uid}|{lines[0]}'
                self._info_lines.append(info_line + '\n')

                # Process wav files
                src_wav = wav_file
                dst_wav = path.join(self.data_dirs[0], f'{self._index:06d}.wav')
                self._wav_paths[0].append(src_wav)
                self._wav_paths[1].append(dst_wav)

                self._index += 1

        # Post settings
        comment = '#index|uid|text\n'
        self._info_lines.insert(0, comment)
        process_func = partial(audio.resample, self.hp)
        self._data_paths = [(self._wav_paths, process_func, 'process')]

        # done

    def _trans_meta_ser(self, lines):
        def _trans_line(line):
            # idx| emo_cls| val| aro| dom| wav_len (s)| wav_name| text
            line = line.split('| ')
            wav_len = float(line[5])
            line.insert(1, str(self.hp.emo_map[line[1]]))  # add emo idx label
            return None if wav_len < self.hp.wav_minlen_s else '| '.join(line)

        res_lines = []
        emo_counter = Counter()
        np.random.shuffle(lines)    # shuffle the samples

        for emo_cls in self.hp.emo_map:
            for line in lines:
                if emo_cls == line[8: 11]:
                    trans_line = _trans_line(line)
                    if not trans_line:
                        continue
                    res_lines.append(trans_line)
                    emo_counter[emo_cls] += 1

        comment = f'#emo map {self.hp.emo_map} -> emo num {dict(emo_counter)}\n'
        comment += '#index| idx| cls| val| aro| dom| wav_len (s)| wav_name| text'
        res_lines.insert(0, comment + '\n')
        return res_lines

    def _make_tfrecord_ser(self, line, *audio_features):
        # index| idx| cls| val| aro| dom| wav_len (s)| wav_name| text
        line = line.split('| ')
        names = ['mel', 'exc_mel', 'mfcc',
                 'hum_scale_mel', 'sine_mel', 'sine_scale_mel']
        num_frames = audio_features[0].shape[-1]  # mel
        audio_features = dict(zip(names, audio_features))
        audio_features = {k: tf_util.bytes_feature(audio_features[k].tobytes())
                          for k in audio_features}
        features = {
            'index': tf_util.int64_feature(int(line[0])),
            'emo_idx': tf_util.int64_feature(int(line[1])),
            'wav_len': tf_util.float_feature(float(line[6])),
            'wav_name': tf_util.bytes_feature(bytes(line[7], 'utf-8')),
            'num_frames': tf_util.int64_feature(num_frames),
        }
        features.update(audio_features)
        res = tf.train.Example(features=tf.train.Features(feature=features))
        return res.SerializeToString()


if __name__ == '__main__':
    dset = BC2013(hp, process=10, thread=10)
    # dset.process_raw()
    # dset.extract_mels('wavs')
    # dset.extract_specs('wavs')
    # dset.extract_excitations('16000_wavs')
    # dset.extract_mfccs('16000_wavs')
    # dset.transform_meta(dset._trans_meta_ser, extra='ser',
    #                     action='transfer meta ser', comment='only use 4 emo cls')
    # dset.make_tfrecords(dset._make_tfrecord_ser, 'train_ser.txt', dst_name='new_ser_tfrs',
                        # feat_names=['80_mels', '80_hum_mels', '39_mfccs',
                                    # '80_hum_scale_mels', '80_sine_mels', '80_sine_scale_mels'],
                        # chops=50, split_key=lambda x: x.split('| ')[1])
