# import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# from utils.debug import debug_print

from modules import custom_layers as cl
from modules import custom_functions as cf
from modules.attention import GSTAttention

from models.tacotron2 import Tacotron2


class Tacotron2EMOGST(Tacotron2):
    def __init__(self, hp, gta=False, name='taco2_emogst'):
        super(Tacotron2EMOGST, self).__init__(hp, gta, name=name)
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
             emo_labels=None,
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
        self.emo_labels = emo_labels

        gst_outputs = tf.tile(gst_outputs, [1, tf.shape(encoder_outputs)[1], 1])
        encoder_outputs = tf.concat([encoder_outputs, gst_outputs], axis=-1)

        # set values for attention layer and batch_timesteps for decoder_cell
        self.atten_layer.set_values_keys(values=encoder_outputs)
        self.decoder_cell.set_batch_timesteps(self.batch_size, tf.shape(encoder_outputs)[1])

        # decoder
        outputs = self.decoder_call(mel_inputs, spec_inputs, spec_lengths)
        return outputs

    def add_emotion_task(self, gst_weights):
        if self.hp.emo_used:
            weights_dim = self.hp.gst_heads * self.hp.gst_tokens
            gst_weights = tf.reshape(gst_weights, [-1, weights_dim])
            self.emo_logits = layers.Dense(4, name='emo_dense')(gst_weights)
        else:
            self.emo_logits = None

    def add_loss(self):
        self.emo_loss = tf.constant(0.0)
        if self.emo_logits is not None:
            loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
            self.emo_loss = loss_fn(self.emo_labels, self.emo_logits)

        super().add_loss()
        self.loss = self.loss + self.emo_loss

    def add_stats(self):
        with tf.variable_scope('stats'):
            if self.hp.emo_used:
                emo_grads = self.optimizer.compute_gradients(self.emo_loss)
                self.emo_grad_norms = [tf.norm(g[0]) for g in emo_grads if g[0] is not None]
                self.emo_grad_norms_max = tf.reduce_max(self.emo_grad_norms)
                tf.summary.scalar('emo_grad_norms_max', self.emo_grad_norms_max)
                tf.summary.histogram('emo_grad_norms', self.emo_grad_norms)
            else:
                self.emo_grad_norms = tf.constant(0)
                self.emo_grad_norms_max = tf.constant(0)

            tf.summary.scalar('emo_loss', self.emo_loss)
            super().add_stats(name='base_stats')
            return tf.summary.merge_all()
