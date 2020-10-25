import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras import backend as K

from utils.infolog import log
# from utils.debug import debug_print
from modules import custom_layers as cl
from modules import custom_functions as cf
from modules.losses import get_mel_loss, get_spec_loss, get_stop_loss


class TacotronBase(keras.Model):
    def __init__(self, hp,
                 encoder,
                 decoder,
                 postnet=None,
                 postcbhg=None,
                 name='TacoBase'):
        super(TacotronBase, self).__init__(name=name)
        self.hp = hp
        self.encoder = encoder
        self.decoder = decoder
        self.postnet = postnet
        self.postcbhg = postcbhg

    def call(self, text_inputs, mel_inputs=None,
             spec_inputs=None, spec_lengths=None, training=None):

        self.training = K.learning_phase() if training is None else training
        self.batch_size = tf.shape(text_inputs)[0]

        # trim the inputs with a length of multiplies of r
        if mel_inputs is not None and spec_inputs is not None:
            mel_inputs, spec_inputs, spec_lengths = cf.trim_inputs(
                self.hp.outputs_per_step, mel_inputs, spec_inputs, spec_lengths)

        # encoder
        encoder_outputs = self.encoder_call(text_inputs)

        # set values for attention layer and text_input_shape for decoder_cell
        self.atten_layer.set_values_keys(values=encoder_outputs)
        self.decoder_cell.set_batch_timesteps(self.batch_size, tf.shape(encoder_outputs)[1])

        # decoder
        outputs = self.decoder_call(mel_inputs, spec_inputs, spec_lengths)
        return outputs

    def encoder_call(self, text_inputs):
        # encoder: takes text(char ids) as input, outupts text embeddings(with mask)
        # [batch, text_time, embeding_dim]
        encoder_outputs = self.encoder(text_inputs, training=self.training)

        self.text_targets = text_inputs
        self.encoder_outputs = encoder_outputs

        return encoder_outputs

    def decoder_call(self, mel_inputs=None, spec_inputs=None, spec_lengths=None):
        hp = self.hp
        training = self.training
        batch_size = self.batch_size

        # decoder: take mels as input, outupts mels, stop_tokens, alignments, and seq_lengths
        # mel [batch, mel_time / r, mel_num * r], stop_token [batch, mel_time / r, r]
        # alignments [batch, mel_time / r, text_time] seq_length [batch]
        decoder_outputs = self.decoder(inputs=mel_inputs,
                                       inputs_lengths=spec_lengths,
                                       batch_size=batch_size,
                                       training=training)
        (decoder_mel, stop_outputs, alignments), _, seq_length_outputs = decoder_outputs

        # output r frame of mels at each time step
        # mel [batch, mel_time, mel_num], stop_token [batch, mel_time, 1]
        decoder_mel = tf.reshape(decoder_mel, shape=[batch_size, -1, hp.num_mels])
        stop_outputs = tf.reshape(stop_outputs, shape=[batch_size, -1])
        if hp.clip_outputs:
            c_min, c_max = hp.clip_min - hp.lower_bound_decay, hp.clip_max
            decoder_mel = tf.clip_by_value(decoder_mel, c_min, c_max)

        # Postnet
        if self.postnet:
            residual_mel = self.postnet(decoder_mel, training)
        else:
            residual_mel = 0.

        # Mel outputs
        mel_outputs = decoder_mel + residual_mel
        if hp.clip_outputs:
            c_min, c_max = hp.clip_min - hp.lower_bound_decay, hp.clip_max
            # mel_outputs = mel_outputs - 0.05
            # mel_outputs = tf.where(mel_outputs < 0.3, mel_outputs - 0.05, mel_outputs)
            mel_outputs = tf.clip_by_value(mel_outputs, c_min, c_max)

        # CBHG convert mel to linear spectrum
        if self.postcbhg:
            self.spec_projection = cl.FrameProjection(hp.num_spec, hp.frame_activation)
            post_outputs = self.postcbhg(mel_outputs, training)
            spec_outputs = self.spec_projection(post_outputs)

            if hp.clip_outputs:
                c_min, c_max = hp.clip_min - hp.lower_bound_decay, hp.clip_max
                # spec_outputs = spec_outputs - 0.05
                # spec_outputs = tf.where(spec_outputs < 0.6, spec_outputs - 0.1, spec_outputs)
                spec_outputs = tf.clip_by_value(spec_outputs, c_min, c_max)
            self.spec_outputs = spec_outputs

        self.mel_targets = mel_inputs
        self.spec_targets = spec_inputs
        self.spec_length_targets = spec_lengths

        self.mel_outputs = mel_outputs
        self.mel_decoder_outputs = decoder_mel
        self.stop_outputs = stop_outputs
        self.alignment_outputs = tf.transpose(alignments, [0, 2, 1])  # [batch, encode_time, decode_time]
        self.seq_length_outputs = seq_length_outputs

        self.all_vars = tf.trainable_variables()

        is_print = True
        log(f'{self.name} Model Dimensions: ', is_print=is_print)
        log('  text embedding:           %d' % self.encoder.embed_output.shape[-1], is_print=is_print)
        log('  encoder  out:             %d' % self.encoder_outputs.shape[-1], is_print=is_print)
        log('  decoder  out:             %d' % mel_outputs.shape[-1], is_print=is_print)
        log('  postcbhg out:             %d' % post_outputs.shape[-1], is_print=is_print)
        log('  linear   out:             %d' % spec_outputs.shape[-1], is_print=is_print)
        log('  Model Parameters       {:.3f} Million.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1000000))

        return mel_outputs, seq_length_outputs

    def add_loss(self, name='loss'):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''

        priority_freq_n = int(2000 / (self.hp.sample_rate * 0.5) * self.hp.num_spec)
        with tf.name_scope(name):
            mel_loss, spec_loss = self.hp.mel_loss, self.hp.spec_loss
            if self.hp.mask_decoder:
                self.before_mel_loss = get_mel_loss(self.mel_targets, self.mel_decoder_outputs,
                                                    self.spec_length_targets, method=mel_loss)
                self.after_mel_loss = get_mel_loss(self.mel_targets, self.mel_outputs,
                                                   self.spec_length_targets, method=mel_loss)
                self.spec_loss = get_spec_loss(self.spec_targets, self.spec_outputs, priority_freq_n,
                                               self.spec_length_targets, method=spec_loss)
                self.stop_loss = get_stop_loss(None, self.stop_outputs, self.hp.outputs_per_step,
                                               self.spec_length_targets, do_mask=True,
                                               pos_weight=self.hp.cross_entropy_pos_weight)
            else:
                self.before_mel_loss = get_mel_loss(self.mel_targets, self.mel_decoder_outputs, method=mel_loss)
                self.after_mel_loss = get_mel_loss(self.mel_targets, self.mel_outputs, method=mel_loss)
                self.spec_loss = get_spec_loss(self.spec_targets, self.spec_outputs, priority_freq_n, method=spec_loss)
                self.stop_loss = get_stop_loss(None, self.stop_outputs, self.hp.outputs_per_step,
                                               self.spec_length_targets)

            self.reg_loss = tf.constant(0.0)
            if self.hp.reg_weight is not None:
                """
                self.reg_vars = [v for v in self.all_vars if not('bias' in v.name or 'atten_cell' in v.name
                                                                 or '_projection' in v.name or 'embeddings' in v.name
                                                                 or 'gru' in v.name or 'lstm' in v.name)]
                """
                self.reg_vars = [v for v in self.all_vars if not('bias' in v.name or '_projection' in v.name or 'embeddings' in v.name)]
                self.reg_loss = self.hp.reg_weight * tf.add_n([tf.nn.l2_loss(v) for v in self.reg_vars], name='reg_loss')

            self.mel_loss = self.before_mel_loss + self.after_mel_loss
            self.loss = self.mel_loss + self.spec_loss + self.stop_loss + self.reg_loss

    def add_optimizer(self, global_step, update_step=True, name='optimizer'):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.
        # Arguments
          global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.name_scope(name):
            hp = self.hp
            if hp.decay_learning_rate:
                self.learning_rate = self.add_learning_rate(hp.initial_learning_rate, global_step)
            else:
                self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            clipped_gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]  # 经常出现梯度爆炸
            clipped_gradients, global_norm = tf.clip_by_global_norm(clipped_gradients, 1.0)

            with tf.control_dependencies(self.updates):
                self.optimize = optimizer.apply_gradients(
                    zip(clipped_gradients, variables),
                    global_step=global_step if update_step else None,
                )

            self.optimizer = optimizer
            self.gradients = gradients
            self.global_gradient_norm = global_norm

    def add_stats(self, name='stats'):
        with tf.name_scope(name):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('reg_loss', self.reg_loss)
            tf.summary.scalar('mel_loss', self.mel_loss)
            tf.summary.scalar('spec_loss', self.spec_loss)
            tf.summary.scalar('stop_loss', self.stop_loss)
            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.histogram('spec_outputs', self.spec_outputs)
            tf.summary.histogram('spec_targets', self.spec_targets)
            tf.summary.histogram('mel_outputs', self.mel_outputs)
            tf.summary.histogram('mel_targets', self.mel_targets)

            self.total_grad_norms = [tf.norm(g) for g in self.gradients]
            self.reg_grad_norms = [tf.norm(g[0]) for g in self.optimizer.compute_gradients(self.reg_loss) if g[0] is not None]
            self.mel_grad_norms = [tf.norm(g[0]) for g in self.optimizer.compute_gradients(self.mel_loss) if g[0] is not None]
            self.spec_grad_norms = [tf.norm(g[0]) for g in self.optimizer.compute_gradients(self.spec_loss) if g[0] is not None]
            self.stop_grad_norms = [tf.norm(g[0]) for g in self.optimizer.compute_gradients(self.stop_loss) if g[0] is not None]
            self.total_grad_norms_max = tf.reduce_max(self.total_grad_norms)
            self.reg_grad_norms_max = tf.reduce_max(self.reg_grad_norms)
            self.mel_grad_norms_max = tf.reduce_max(self.mel_grad_norms)
            self.spec_grad_norms_max = tf.reduce_max(self.spec_grad_norms)
            self.stop_grad_norms_max = tf.reduce_max(self.stop_grad_norms)

            tf.summary.scalar('global_grad_norm', self.global_gradient_norm)
            tf.summary.scalar('total_grad_norms_max', self.total_grad_norms_max)
            tf.summary.scalar('reg_grad_norms_max', self.reg_grad_norms_max)
            tf.summary.scalar('mel_grad_norms_max', self.mel_grad_norms_max)
            tf.summary.scalar('spec_grad_norms_max', self.spec_grad_norms_max)
            tf.summary.scalar('stop_grad_norms_max', self.stop_grad_norms_max)

            tf.summary.histogram('total_grad_norms', self.total_grad_norms)
            tf.summary.histogram('reg_grad_norms', self.reg_grad_norms)
            tf.summary.histogram('mel_grad_norms', self.mel_grad_norms)
            tf.summary.histogram('spec_grad_norms', self.spec_grad_norms)
            tf.summary.histogram('stop_grad_norms', self.stop_grad_norms)
            return tf.summary.merge_all()

    def add_learning_rate(self, init_lr, global_step):
        # Noam scheme from tensor2tensor:
        warmup_steps = 4000.0
        step = tf.cast(global_step + 1, dtype=tf.float32)
        return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
