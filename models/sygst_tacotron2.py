import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# from utils.debug import debug_print

from modules import custom_layers as cl
from modules import custom_functions as cf
from modules.attention import GSTAttention

from models.tacotron2 import Tacotron2


class Tacotron2SYGST(Tacotron2):
    def __init__(self, hp, gta=False, name='taco2_sygst'):
        super(Tacotron2SYGST, self).__init__(hp, gta, name=name)
        self.reference_encoder = cl.ReferenceEncoder(
            channels=hp.reference_channels,
            rnn_units=hp.reference_rnn_units,
            output_units=hp.emotion_embedding_units)
        self.gst_attention = GSTAttention(
            num_heads=hp.gst_heads,
            num_tokens=hp.gst_tokens,
            gst_units=hp.gst_units,
            attention_units=hp.gst_atten_units,
            attention_type=hp.gst_atten_type,
            activation=hp.gst_activation,
            trainable=hp.gst_trainable)

    def call(self,
             text_inputs,
             mel_inputs=None,
             spec_inputs=None,
             spec_lengths=None,
             ref_inputs=None,
             ref_lengths=None,
             arousal_labels=None,
             valence_labels=None,
             atten_weights_ph=None,
             training=None):

        self.training = K.learning_phase() if training is None else training
        self.batch_size = tf.shape(text_inputs)[0]

        # trim the inputs with a length of multiplies of r
        if mel_inputs is not None and spec_inputs is not None:
            mel_inputs, spec_inputs, spec_lengths = cf.trim_inputs(
                self.hp.outputs_per_step, mel_inputs, spec_inputs, spec_lengths)

        # set reference to mel if it is not given
        if ref_inputs is None:
            ref_inputs = mel_inputs
            ref_lengths = spec_lengths

        # encoder
        encoder_outputs = self.encoder_call(text_inputs)

        # reference encoder
        ref_outputs = None
        if ref_inputs is not None:  # 之前没有传递training参数, 导致bn层参数梯度nan
            ref_outputs = self.reference_encoder(ref_inputs, x_length=ref_lengths, training=training)
        gst_outputs = self.gst_attention(ref_outputs, atten_weights_ph=atten_weights_ph, training=training)

        self.ref_outputs = ref_outputs  # [N, ref_output_units]
        self.gst_outputs = gst_outputs  # [N, 1, gst_units]
        self.gst_weights = self.gst_attention.atten_weights  # [N, gst_heads, 1, gst_tokens]

        self.add_emotion_task(self.gst_weights)
        self.arousal_labels = arousal_labels
        self.valence_labels = valence_labels

        gst_outputs = tf.tile(gst_outputs, [1, tf.shape(encoder_outputs)[1], 1])
        encoder_outputs = tf.concat([encoder_outputs, gst_outputs], axis=-1)

        # set values for attention layer and batch_timesteps for decoder_cell
        self.atten_layer.set_values_keys(values=encoder_outputs)
        self.decoder_cell.set_batch_timesteps(self.batch_size, tf.shape(encoder_outputs)[1])

        # decoder
        outputs = self.decoder_call(mel_inputs, spec_inputs, spec_lengths)
        return outputs

    def add_emotion_task(self, gst_weights):
        units, emo_loss = self.hp.emo_output_units, self.hp.emo_loss
        units = 1 if emo_loss in ['mae', 'mse', 'sigmoid'] else units

        if self.hp.emo_used:
            arousal_weights, valence_weights = tf.split(gst_weights, 2, axis=1)
            weights_dim = np.prod(arousal_weights.shape[1:])
            arousal_weights = tf.reshape(arousal_weights, [-1, weights_dim])
            valence_weights = tf.reshape(valence_weights, [-1, weights_dim])
            self.arousal_logits = layers.Dense(units, name='aro_dense')(arousal_weights)
            self.valence_logits = layers.Dense(units, name='val_dense')(valence_weights)
        else:
            self.arousal_logits = None
            self.valence_logits = None

    def add_loss(self):
        emo_loss = self.hp.emo_loss
        if emo_loss in ['mae', 'mse']:
            loss_fn = keras.losses.get(emo_loss)
        elif emo_loss == 'sigmoid':
            loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        elif emo_loss == 'softmax':
            loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        else:
            raise ValueError(f'The emo_loss={emo_loss} is not valid')

        self.arousal_loss = tf.constant(0.0)
        self.valence_loss = tf.constant(0.0)
        if self.arousal_logits is not None:
            self.arousal_loss = loss_fn(self.arousal_labels, self.arousal_logits)
            self.valence_loss = loss_fn(self.valence_labels, self.valence_logits)
        self.emo_loss = self.arousal_loss + self.valence_loss

        super().add_loss()
        self.loss = self.loss + self.emo_loss

    def add_stats(self, name='stats'):
        with tf.name_scope(name):
            if self.hp.emo_used:
                aro_grads = self.optimizer.compute_gradients(self.arousal_loss)
                val_grads = self.optimizer.compute_gradients(self.valence_loss)
                self.aro_grad_norms = [tf.norm(g[0]) for g in aro_grads if g[0] is not None]
                self.val_grad_norms = [tf.norm(g[0]) for g in val_grads if g[0] is not None]
                self.aro_grad_norms_max = tf.reduce_max(self.aro_grad_norms)
                self.val_grad_norms_max = tf.reduce_max(self.val_grad_norms)

                tf.summary.scalar('aro_grad_norms_max', self.aro_grad_norms_max)
                tf.summary.scalar('val_grad_norms_max', self.val_grad_norms_max)
                tf.summary.histogram('aro_grad_norms', self.aro_grad_norms)
                tf.summary.histogram('val_grad_norms', self.val_grad_norms)
            else:
                self.aro_grad_norms = tf.constant(0)
                self.val_grad_norms = tf.constant(0)
                self.aro_grad_norms_max = tf.constant(0)
                self.val_grad_norms_max = tf.constant(0)

            tf.summary.scalar('aro_loss', self.arousal_loss)
            tf.summary.scalar('val_loss', self.valence_loss)
            super().add_stats(name='base_stats')
        return tf.summary.merge_all()
