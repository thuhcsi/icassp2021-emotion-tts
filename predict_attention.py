import re
import os
import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input

from text import text_to_sequence

from sygst_hparams import hp as sygst_hp
from embjoint_hparams import hp as embgst_joint_hp
from ser.hparams import hp as ser_hp
from emogst_hparams import hp as emogst_hp
from models.sygst_tacotron2 import Tacotron2SYGST
from models.embgst_tacotron2_joint import Tacotron2EMBGSTJoint
from models.emogst_tacotron2 import Tacotron2EMOGST

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

map_model = {'sygst': Tacotron2SYGST, 'embgst_joint': Tacotron2EMBGSTJoint, 'emogst': Tacotron2EMOGST}
map_hp = {'sygst': sygst_hp, 'embgst_joint': embgst_joint_hp, 'emogst': emogst_hp}


class AttentionPredictor:
    def __init__(self, model_name='sygst'):
        assert model_name in ['sygst', 'embgst', 'embgst_joint', 'emogst']

        self.hp = map_hp[model_name]
        self.model = map_model[model_name](self.hp) if model_name != 'embgst' else map_model[model_name](self.hp, ser_hp)
        self.model_name = model_name
        self.cleaner_names = [x.strip() for x in self.hp.cleaners.split(',')]

        # build model
        with tf.name_scope('model'):   # 只能和训练时的scope一致
            d = self.hp.emotion_embedding_units
            self.text_inputs = Input([None], dtype=tf.int32, name='text_inputs')
            self.mel_inputs = Input([None, self.hp.num_mels], dtype=tf.float32, name='mel_inputs')
            self.spec_lengths = Input([], dtype=tf.int32, name='spec_lengths')
            self.aro_embed = Input([d], dtype=tf.float32, name='aro_embed')
            self.val_embed = Input([d], dtype=tf.float32, name='val_embed')

            call_fn_kwargs = {'mel_inputs': self.mel_inputs,
                              'spec_lengths': self.spec_lengths,
                              'training': False}
            if model_name == 'embgst':
                call_fn_kwargs.update(arousal_embedding=self.aro_embed,
                                      valence_embedding=self.val_embed)
            self.model(self.text_inputs, **call_fn_kwargs)

    def load(self, ckpt_path):
        print('Loading checkpoint: %s' % ckpt_path)
        self.eval_step = re.search(r'ckpt-(\d+)', ckpt_path).group(1)
        self.session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.session, ckpt_path)

    def predict(self, mel_inputs=None, spec_lengths=None, aro_embed=None, val_embed=None):
        seq = text_to_sequence('hello', self.cleaner_names)
        feed_dict = {self.text_inputs: [np.asarray(seq, dtype=np.int32)]}
        if mel_inputs is not None:
            assert spec_lengths is not None
            mel_inputs = np.expand_dims(mel_inputs, 0).astype(np.float32)
            spec_lengths = np.expand_dims(spec_lengths, 0).astype(np.int32)
            feed_dict.update({self.mel_inputs: mel_inputs, self.spec_lengths: spec_lengths})
            if self.model_name in ['sygst', 'emogst']:
                attention_outputs = self.model.gst_weights
            elif self.model_name == 'embgst_joint':
                attention_outputs = [self.model.aro_weights, self.model.val_weights]
            else:
                raise ValueError('when mel_inputs is not None, model must be sygst or embgst_joint')
        else:
            assert aro_embed is not None or val_embed is not None
            if aro_embed is not None:
                aro_embed = np.expand_dims(aro_embed, 0)
                feed_dict.update({self.arousal_embedding: aro_embed.astype(np.float32)})
                attention_outputs = self.model.aro_weights
            else:
                val_embed = np.expand_dims(val_embed, 0)
                feed_dict.update({self.valence_embedding: val_embed.astype(np.float32)})
                attention_outputs = self.model.val_weights

        attention_weights = self.session.run(attention_outputs, feed_dict=feed_dict)
        return attention_weights


def process_fold(args, model, ref_path, output_path, emo_type='arousal', max_items=50):
    atten_list = []
    # ref_names = [os.path.join(ref_path, name) for name in sorted(os.listdir(ref_path))]
    ref_names = [os.path.join(ref_path, name) for name in os.listdir(ref_path)]
    ref_names = ref_names[:max_items] if max_items is not None else ref_names

    for ref_name in ref_names:
        ref_feature = np.load(ref_name)
        if args.model_name in ['sygst', 'embgst_joint', 'emogst']:
            ref_len = ref_feature.shape[0]
            # if ref_len < 250 or ref_len > 1000:
            #     continue
            atten_weight = model.predict(mel_inputs=ref_feature, spec_lengths=ref_len)
            if args.model_name == 'embgst_joint':
                atten_weight = atten_weight[0] if args.emo_type == 'arousal' else atten_weight[1]
        else:
            assert emo_type in ['arousal', 'valence']
            if emo_type == 'arousal':
                atten_weight = model.predict(aro_embed=ref_feature)
            else:
                atten_weight = model.predict(val_embed=ref_feature)
        atten = np.squeeze(atten_weight, 0)  # [num_heads, 1, num_tokens]
        atten_list.append(atten)
    atten_list_np = np.array(atten_list)
    avg_atten = np.mean(atten_list_np, axis=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, avg_atten)
    print(f'Process finished for {args.model_name} {emo_type} with shape: {atten_list_np.shape}')


def process_arousal(args, model):
    for i in range(2):
        ref_path = args.model_name + f'_emo_data/emo2d_mel_npys/arousal{i}'
        output_path = args.model_name + f'_emo_data/emo2d_mel_gst_weights/arousal{i}.npy'
        if args.model_name == 'embgst':
            ref_path = ref_path.replace('_mel_', '_embed_')
            output_path = output_path.replace('_mel_', '_embed_')
        process_fold(args, model, ref_path, output_path, 'arousal')


def process_valence(args, model):
    for i in range(2):
        ref_path = args.model_name + f'_emo_data/emo2d_mel_npys/valence{i}'
        output_path = args.model_name + f'_emo_data/emo2d_mel_gst_weights/valence{i}.npy'
        if args.model_name == 'embgst':
            ref_path = ref_path.replace('_mel_', '_embed_')
            output_path = output_path.replace('_mel_', '_embed_')
        process_fold(args, model, ref_path, output_path, 'valence')


def process_emotion(args, model):
    for i in range(4):
        ref_path = args.model_name + f'_emo_data/emo_mel_npys/emo{i}'
        output_path = args.model_name + f'_emo_data/emo_gst_weights/emo{i}.npy'
        process_fold(args, model, ref_path, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', default='sygst')
    parser.add_argument('--ckpt_step', '-c', default=None)
    args = parser.parse_args()

    assert args.model_name in ['sygst', 'embgst', 'embgst_joint', 'emogst']

    ckpt_path = args.model_name + f'_emo_data/ckpts/model.ckpt-{args.ckpt_step}'
    model = AttentionPredictor(args.model_name)
    model.load(ckpt_path)

    if args.model_name == 'emogst':
        process_emotion(args, model)
    else:
        process_arousal(args, model)
        process_valence(args, model)


if __name__ == '__main__':
    main()
