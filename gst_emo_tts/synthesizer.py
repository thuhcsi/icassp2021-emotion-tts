import re
import textwrap
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.layers import Input

from utils import audio, plot

from ser.hparams import hp as ser_hp
from taco2_hparams import hp as taco2_hp
from sygst_hparams import hp as sygst_hp
from emogst_hparams import hp as emogst_hp
from embjoint_hparams import hp as embgst_joint_hp

from models.tacotron2 import Tacotron2
from models.sygst_tacotron2 import Tacotron2SYGST
from models.emogst_tacotron2 import Tacotron2EMOGST
from models.embgst_tacotron2_joint import Tacotron2EMBGSTJoint

tf.compat.v1.logging.set_verbosity(40)   # Only print error infos

map_model = {'taco2': Tacotron2, 'sygst': Tacotron2SYGST,
             'emogst': Tacotron2EMOGST, 'embgst_joint': Tacotron2EMBGSTJoint}
map_hp = {'taco2': taco2_hp, 'sygst': sygst_hp,
          'emogst': emogst_hp, 'embgst_joint': embgst_joint_hp}


class Synthesizer:
    def __init__(self, use_gta=False, use_ref=False, use_att=True, model_name='taco2'):

        assert model_name in ['taco2', 'sygst', 'embgst', 'emogst', 'embgst_joint']

        self.use_gta = use_gta  # whether using ground truth alignment
        self.use_ref = use_ref  # whether using reference mel
        self.use_att = use_att  # whether using attention weights
        self.hp = map_hp[model_name]
        self.model = map_model[model_name](self.hp, use_gta) if model_name != 'embgst_joint' else map_model[model_name](embgst_joint_hp, ser_hp, use_gta)
        self.model_name = model_name

        # build model
        with tf.name_scope('model'):
            h, t = self.hp.gst_heads, self.hp.gst_tokens
            self.text_inputs = Input([None], dtype=tf.int32, name='text_inputs')
            self.mel_inputs = Input([None, self.hp.num_mels], dtype=tf.float32, name='mel_inputs')
            self.mel_lengths = Input([], dtype=tf.int32, name='mel_lengths')
            self.ref_inputs = Input([None, self.hp.num_mels], dtype=tf.float32, name='ref_inputs')
            self.ref_lengths = Input([], dtype=tf.int32, name='ref_lengths')
            self.aro_weights_ph = Input([h, 1, t], dtype=tf.float32, name='arousal_weitght_ph')
            self.val_weights_ph = Input([h, 1, t], dtype=tf.float32, name='valence_weitght_ph')
            self.atten_weights_ph = Input([h, 1, t], dtype=tf.float32, name='attention_weights_ph')

            call_fn_kwargs = {}
            if use_gta:
                assert not use_ref
                call_fn_kwargs.update(mel_inputs=self.mel_inputs,
                                      spec_lengths=self.mel_lengths)
            if use_ref:
                assert not use_att and model_name != 'taco2'
                call_fn_kwargs.update(ref_inputs=self.ref_inputs,
                                      ref_lengths=self.ref_lengths)
            if use_att:
                if model_name in ['sygst', 'emogst']:
                    call_fn_kwargs.update(atten_weights_ph=self.atten_weights_ph)
                elif model_name in ['embgst', 'embgst_joint']:
                    call_fn_kwargs.update(aro_weights_ph=self.aro_weights_ph,
                                          val_weights_ph=self.val_weights_ph)
            self.model_call_fn_kwargs = call_fn_kwargs
            self.model(self.text_inputs, training=False, **call_fn_kwargs)

        # outputs
        model = self.model if self.model_name != 'embgst_joint' else self.model.tts_model
        self.seq_length_outputs = model.seq_length_outputs
        self.mel_outputs = model.mel_outputs
        self.spec_outputs = model.spec_outputs
        self.wav_outputs = audio.inv_spectrogram_tensorflow(self.hp, self.spec_outputs)
        self.alignment_outputs = model.alignment_outputs

    def load(self, ckpt_path):
        self.eval_step = re.search(r'ckpt-(\d+)', ckpt_path).group(1)
        self.session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.session, ckpt_path)

    def synthesize(self, text_seqs, texts, output_path,
                   mel_inputs=None, mel_lengths=None,
                   ref_inputs=None, ref_lengths=None,
                   atten_weights=None, aro_weights=None, val_weights=None):

        feed_dict = {self.text_inputs: text_seqs}
        if mel_inputs is not None:
            feed_dict.update({self.mel_inputs: mel_inputs,
                              self.mel_lengths: mel_lengths})
        if ref_inputs is not None:
            feed_dict.update({self.ref_inputs: ref_inputs,
                              self.ref_lengths: ref_lengths})
        if aro_weights is not None and val_weights is not None:
            feed_dict.update({self.aro_weights_ph: aro_weights,
                              self.val_weights_ph: val_weights})
        if atten_weights is not None:
            feed_dict.update({self.atten_weights_ph: atten_weights})

        self.now_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        lens, wavs, mels, specs, aligns = self.session.run([self.seq_length_outputs,
                                                            self.wav_outputs,
                                                            self.mel_outputs,
                                                            self.spec_outputs,
                                                            self.alignment_outputs],
                                                           feed_dict=feed_dict)
        self.post_process(output_path, texts, lens, wavs, mels, specs, aligns)

    def post_process(self, output_path, texts, lens, wavs, mels, specs, aligns):

        zipped_inputs = zip(output_path, texts, lens, wavs, mels, specs, aligns)
        for path, text, mel_len, wav, mel, spec, align in zipped_inputs:
            wav = audio.inv_preemphasis(self.hp, wav)
            end_point = audio.find_endpoint(self.hp, wav)
            # end_point = wav.shape[0]
            # end_point = int((mel_len * self.hp.hop_ms / 1000) * self.hp.sample_rate)
            wav = wav[:end_point]
            mel_len = int(end_point / (self.hp.hop_ms / 1000 * self.hp.sample_rate)) + 1
            pathes = [path + suffix for suffix in ['.wav', '-mel.png', '-spec.png', '-align.png']]
            wav_path, mel_path, spec_path, align_path = pathes
            title = f'{self.model_name}, {self.eval_step}, {self.now_time}'
            info = '\n'.join(textwrap.wrap(text, 70, break_long_words=False))
            plot.plot_alignment(align[:, : mel_len], align_path, info, title)
            plot.plot_mel(mel[: mel_len, :], mel_path, info, title)
            plot.plot_mel(spec[: mel_len, :], spec_path, info, title)
            audio.save_wav(self.hp, wav, wav_path)
