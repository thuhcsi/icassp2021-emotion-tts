import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.compat.v1.logging import warn
from tensorflow.python.keras.utils import tf_utils
# from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauMonotonicAttention

from collections import namedtuple


class BaseAttention(layers.Layer):
    def __init__(self,
                 units,
                 q_layer=None,
                 k_layer=None,
                 v_layer=None,
                 o_layer=None,
                 o_units=None,
                 a_activation=None,
                 o_activation=None,
                 probability_fn=None,
                 score_mask_value=None,
                 name='base_attention'):
        super(BaseAttention, self).__init__(name=name)
        # self.supports_masking = True

        self.units = units       # attention units
        self.o_units = o_units   # output units
        self.a_activaiton = a_activation
        self.o_activation = o_activation

        self._probability_fn = probability_fn or (lambda score, state: tf.nn.softmax(score))
        self.score_mask_value = -np.inf if score_mask_value is None else score_mask_value
        self.probability_fn = self._probability_fn

        self._batch_size = None
        self._time_steps = None
        self._init_state = None
        self._dim_v = None

        self._set_value_key = False

        def _process_layer(layer, units, activation, name):
            if layer is None:
                L = layers.Lambda(tf.identity, name=name)
                L.supports_masking = True
            elif layer in ['linear', 'dense']:
                L = layers.Dense(units, activation, use_bias=False, name=name)
            elif callable(layer):
                L = layer
            else:
                raise ValueError(f'the layer: {layer} argument is not valid')
            return L

        self.q_layer = _process_layer(q_layer, units, a_activation, name='q_layer')
        self.k_layer = _process_layer(k_layer, units, a_activation, name='k_layer')
        self.v_layer = _process_layer(v_layer, units, a_activation, name='v_layer')
        self.o_layer = _process_layer(o_layer, o_units, o_activation, name='o_layer')

    def set_values_keys(self, values, keys=None, key_lengths=None):
        """Set values, keys and probably a mask for probability_fn
        # Notes
            This usually is used for time-step gradually decoding at decoder
        """
        if self._set_value_key:
            return
        assert values is not None, 'Must set values with tf.Tensor instead of None'

        keys = values if keys is None else keys

        def _process_mask(mask_value, data=None):
            # The mask of values and keys should be equal actually
            if hasattr(keys, '_keras_mask') and values._keras_mask is not None:
                mask = keys._keras_mask
            elif key_lengths is not None:
                mask = tf.sequence_mask(key_lengths, maxlen=tf.shape(values)[1])
            else:
                mask = tf.constant(True, shape=[1, 1], dtype=bool)
                warn(RuntimeWarning('Both values._keras_mask and key_legnths are None'))

            one_value = tf.ones_like(mask, dtype=tf.float32)
            mask_value = one_value * mask_value

            if data is not None:
                # multihead attention with a extra 'head' axis
                if mask.shape.ndims != data.shape.ndims:
                    extra_dims = data.shape.ndims - mask.shape.ndims
                    for _ in range(extra_dims):
                        mask = tf.expand_dims(mask, axis=1)
                        mask_value = tf.expand_dims(mask_value, axis=1)
                mask = tf.broadcast_to(mask, tf.shape(data))
                mask_value = tf.broadcast_to(mask_value, tf.shape(data))
                mask = tf.where(mask, data, mask_value)
            else:
                mask = tf.where(mask, one_value, mask_value)
            return mask

        self.kv_mask = tf.expand_dims(_process_mask(0), axis=2)
        self.k = keys * self.kv_mask
        self.v = values * self.kv_mask

        def probability_fn(score, prev_state=None):
            return self._probability_fn(_process_mask(self.score_mask_value, score), prev_state)
        self.probability_fn = probability_fn

        self._batch_size = tf.shape(self.v)[0]
        self._time_steps = tf.shape(self.v)[1]
        self._dim_v = self.v.shape.as_list()[2]  # 必须用常量值, 否则atten_rnn无法build

        self._set_value_key = True

    def call(self, query, state=None):
        """
        # Arguments
            query: current timestep attention query
            state: previous attention state, its meaning dependend on the
                definition from subclasses. it usually is a namedtuple object.
        # Returns
            context: current timestep attention context vector
            next_state: next timestep attention state. Note, current time step
                socres must be an attribute of next_state

        """
        raise NotImplementedError('Abstract method')

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def time_steps(self):
        return self._time_steps

    @property
    def dim_v(self):
        return self._dim_v


class LocationSensitiveAttention(BaseAttention):
    _State = namedtuple('State', ['context', 'alignments', 'accum_alignments'])

    def __init__(self,
                 units,
                 location_filters=32,
                 location_kernel_size=31,
                 synthesis_constraint=True,
                 synthesis_win_size=6,
                 synthesis_softmax_temp=1.0,
                 name='location_sensitive_attention'):
        """
        # Arguments
            synthesis_constraint: whether use window constraint for computing
                attention score, if True and not training, constraint will be used
        """
        super(LocationSensitiveAttention, self).__init__(units,
                                                         q_layer='linear',
                                                         k_layer='linear',
                                                         name=name)
        self.location_filters = location_filters
        self.location_kernel_size = location_kernel_size
        self.synthesis_constraint = synthesis_constraint
        self.synthesis_win_size = synthesis_win_size
        self.synthesis_softmax_temp = synthesis_softmax_temp

        self.location_conv_layer = layers.Conv1D(location_filters,
                                                 location_kernel_size,
                                                 padding='same', name='conv')
        self.location_dense_layer = layers.Dense(units, use_bias=True, name='conv_dense')  # add a bias before tanh
        self.mlp_layer = layers.Dense(1, use_bias=False, name='mlp_dense')

    def call(self, query, state, training=None):
        """
        # Arguments
            query: current timestep attention query
            state: accumulated alignments
            training: used for determine whether use synthesis constraint
        """
        if not self._set_value_key:
            raise RuntimeError('attention has not been set values now, you can not call it')

        if training is None:
            training = K.learning_phase()

        q = self.q_layer(query)   # [batch, units]
        k = self.k_layer(self.k)  # [batch, max_time, units]
        v = self.v_layer(self.v)  # [batch, max_time, embedding_dim]

        # concat accum_alignments and alignments. from nvidia pytorch tacotron2
        accum_alignments = tf.stack([state.alignments, state.accum_alignments],
                                    axis=-1, name='alignments_concat')

        expand_query = tf.expand_dims(q, axis=1)  # [batch, 1, units]
        location_query = self.location_conv_layer(accum_alignments)  # [batch, max_time, atten_filters]
        location_query = self.location_dense_layer(location_query)   # [batch, max_time, units]

        energies = self.mlp_layer(tf.nn.tanh(expand_query + location_query + k))  # [batch, max_time, 1]
        energies = tf.squeeze(energies, axis=2)   # [batch, max_time]
        energies = self._energies_constriant(energies, state.alignments, training)
        scores = self.probability_fn(energies)      # probability_fn with mask [batch, max_time]
        context = tf.einsum('ai,aik->ak', scores, v)    # [batch, embedding_dim]

        next_state = self.State(context=context,
                                alignments=scores,
                                accum_alignments=scores + state.accum_alignments)
        return context, next_state

    def _energies_constriant(self, energies, prev_alignments, training):
        def _true_fn():
            prev_max_step = tf.argmax(prev_alignments, axis=-1)
            max_steps = tf.shape(energies)[-1]
            win_mask = tf.logical_xor(
                tf.sequence_mask(prev_max_step - np.ceil(self.synthesis_win_size / 2).astype(np.int64), max_steps),
                tf.sequence_mask(prev_max_step + np.floor(self.synthesis_win_size / 2).astype(np.int64), max_steps)
            )
            mask_paddings = -np.inf * tf.ones_like(energies)
            return tf.where(win_mask, energies, mask_paddings)

        def _false_fn():
            return tf.identity(energies)
        energies = energies * self.synthesis_softmax_temp  # temp>1 can make the alignment brighter
        pred = tf.logical_and(self.synthesis_constraint, tf.logical_not(training))
        return tf_utils.smart_cond(pred, _true_fn, _false_fn)

    @property
    def State(self):
        return self._State

    @property
    def init_state(self):
        assert self._set_value_key
        if self._init_state:
            return self._init_state

        self._init_state = self.State(
            context=tf.zeros(shape=(self.batch_size, self.dim_v)),
            alignments=tf.zeros(shape=(self.batch_size, self.time_steps)),
            accum_alignments=tf.zeros(shape=(self.batch_size, self.time_steps)),
        )
        return self._init_state


class StepwiseMonotonicAttention(BaseAttention):
    _State = namedtuple('State', ['context', 'alignments'])

    def __init__(self,
                 units,
                 normalize=True,
                 sigmoid_noise=2.0,
                 sigmoid_noise_seed=None,
                 score_bias_init=3.5,
                 mode="parallel",
                 name="stepwise_monotonic_attention"):

        assert mode in ['parallel', 'hard']
        assert isinstance(sigmoid_noise, (float, int))
        super(StepwiseMonotonicAttention, self).__init__(units,
                                                         q_layer='linear',
                                                         k_layer='linear',
                                                         name=name)
        self.normalize = normalize
        self.sigmoid_noise = sigmoid_noise
        self.sigmoid_noise_seed = sigmoid_noise_seed
        self.score_bias_init = score_bias_init
        self.mode = mode

        self._probability_fn = self._stepwise_monotonic_probability_fn
        self.probability_fn = self._probability_fn

    def build(self, input_shape):
        self.attention_v = self.add_weight(
            name='attention_v',
            shape=[self.units],
            trainable=True
        )
        self.score_bias = self.add_weight(
            name='attention_score_bias',
            initializer=keras.initializers.constant(self.score_bias_init, dtype=tf.float32),
            trainable=True
        )
        if self.normalize:
            self.attention_g = self.add_weight(
                name='attention_g',
                initializer=keras.initializers.constant((1. / self.units) ** 0.5, dtype=tf.float32),
                trainable=True
            )
            self.attention_b = self.add_weight(
                name='attention_b',
                shape=[self.units],
                initializer=keras.initializers.zeros(),
                trainable=True
            )
        self.built = True

    def call(self, query, state):
        """
        # Arguments
            query: current timestep attention query
            state: previous timestep alignments
        """
        if not self._set_value_key:
            raise RuntimeError('attention has not been set values now, you can not call it')

        q = self.q_layer(query)   # [batch, units]
        k = self.k_layer(self.k)  # [batch, max_time, units]
        v = self.v_layer(self.v)  # [batch, max_time, embedding_dim]
        expand_query = tf.expand_dims(q, axis=1)  # [batch, 1, units]

        if self.normalize:
            normed_v = self.attention_g * tf.nn.l2_normalize(self.attention_v)
            energies = tf.einsum('j,aij->ai', normed_v,
                                 tf.nn.tanh(expand_query + k + self.attention_b))
        else:
            energies = tf.einsum('j,aij->ai', self.attention_v,
                                 tf.nn.tanh(expand_query + k))
        energies += self.score_bias  # [batch, max_time]
        scores = self.probability_fn(energies, state.alignments)     # [batch, max_time]
        context = tf.einsum('ai,aik->ak', scores, v)                 # [batch, embedding_dim]

        next_state = self.State(context=context, alignments=scores)

        return context, next_state

    def monotonic_stepwise_attention(self, p_choose_i, previous_attention):
        # p_choose_i, previous_alignments, previous_score: [batch_size, memory_size]
        # p_choose_i: probability to keep attended to the last attended entry i
        if self.mode == "parallel":
            pad = tf.zeros([tf.shape(p_choose_i)[0], 1], dtype=p_choose_i.dtype)
            attention = previous_attention * p_choose_i + tf.concat(
                [pad, previous_attention[:, :-1] * (1.0 - p_choose_i[:, :-1])], axis=1)
        elif self.mode == "hard":
            # Given that previous_alignments is one_hot
            move_next_mask = tf.concat([tf.zeros_like(previous_attention[:, :1]), previous_attention[:, :-1]], axis=1)
            stay_prob = tf.reduce_sum(p_choose_i * previous_attention, axis=1)  # [B]
            attention = tf.where(stay_prob > 0.5, previous_attention, move_next_mask)
        return attention

    def _stepwise_monotonic_probability_fn(self, score, previous_alignments):
        """
        # Arguments
            score: Unnormalized attention scores, shape `[batch_size, alignments_size]`
            previous_alignments: Previous attention distribution, shape `[batch_size, alignments_size]`
        """
        if self.sigmoid_noise > 0:
            noise = tf.random_normal(tf.shape(score), dtype=score.dtype, seed=self.sigmoid_noise_seed)
            score += self.sigmoid_noise * noise
        if self.mode == "hard":
            # When mode is hard, use a hard sigmoid
            p_choose_i = tf.cast(score > 0, score.dtype)
        else:
            p_choose_i = tf.sigmoid(score)
        alignments = self.monotonic_stepwise_attention(p_choose_i, previous_alignments)
        return alignments

    @property
    def State(self):
        return self._State

    @property
    def init_state(self):
        if not self._set_value_key:
            return None
        if self._init_state:
            return self._init_state

        init_context = tf.zeros(shape=(self.batch_size, self.dim_v))
        init_alignments = tf.one_hot(tf.zeros((self.batch_size,), dtype=tf.int32),
                                     self.time_steps)
        self._init_state = self.State(init_context, init_alignments)
        return self._init_state


class MultiHeadAttention(BaseAttention):
    def __init__(self,
                 num_heads,
                 attention_units,
                 attention_type='mlp',
                 attention_drop=None,
                 v_layer='linear',
                 output_units=None,
                 activation=None,
                 probability_fn=None,
                 name='multi_head_attention'):
        super(MultiHeadAttention, self).__init__(units=attention_units,
                                                 q_layer='linear',
                                                 k_layer='linear',
                                                 v_layer=v_layer,
                                                 o_layer='linear' if output_units else None,
                                                 o_units=output_units,
                                                 a_activation=activation,
                                                 o_activation=activation,
                                                 probability_fn=probability_fn,
                                                 name=name)
        self.num_heads = num_heads
        self.attention_units = attention_units
        self.attention_type = attention_type
        self.output_units = output_units

        self.attention_drop = layers.Dropout(attention_drop, name='drop') if attention_drop else None

    def call(self, x, y, positions=None, atten_weights_ph=None, training=None):
        '''Computes multihead attention
        # Arguments
            x: used to compute query with shape [batch, len_q, dim_q]
            y: used to compute key and value with shape [batch, len_k, dim_k]
            positions: not suppourt now
        # Returns
            output: a tensor with shape [batch, len_q, attention_units] that
                computed by multihead attention

        '''
        # Project q k v to have same channels(attention units)
        q = self.q_layer(x)       # [batch, len_q, attention_units]
        k = self.k_layer(y)       # [batch, len_k, attention_units]
        v = self.v_layer(y)       # [batch, len_k, attention_units]

        # Split q k v into heads with dim = attention_units // num_heads
        qs = self.split_heads(q)  # [batch, num_heads, len_q, dim]
        ks = self.split_heads(k)  # [batch, num_heads, len_k, dim]
        vs = self.split_heads(v)  # [batch, num_heads, len_k, dim]

        # Compute attention weights
        assert self.attention_type in ['mlp', 'dot']
        # [batch, num_heads, len_q, len_k]
        if self.attention_type == 'mlp':
            atten_weights = self.mlp_attention(qs, ks)
        else:
            atten_weights = self.dot_attention(qs, ks)
        if self.attention_drop:
            atten_weights = self.attention_drop(atten_weights, training)

        if atten_weights_ph is not None:    # used for emotional gst tts inference
            atten_weights = atten_weights_ph

        self.atten_weights = atten_weights  # used for emotional gst tts

        # [batch, num_heads, len_q, dim]
        output = tf.einsum('abij,abjk->abik', atten_weights, vs)
        output = self.combine_heads(output)  # [batch, len_q, attention_units]
        output = self.o_layer(output)        # [batch, len_q, attention_units]
        return output

    def split_heads(self, x):
        '''Split the channels into multiple heads
        # Arguments
            x: A tensor with shape [batch, length, attention_units]
        # Returns
             Tensors with shape [batch, num_heads, length, attention_units/num_heads]
        '''
        x = tf.stack(tf.split(x, self.num_heads, axis=-1), axis=1, name='split_heads')
        return x

    def combine_heads(self, x):
        '''Combine the heads
        # Arguments
            x: A tensor with shape [batch, num_heads, length, attention_units/num_heads]
        # Returns
             Tensors with shape [batch, length, attention_units]
        '''
        x = tf.concat(tf.unstack(x, axis=1), axis=-1, name='combine_heads')
        return x

    def dot_attention(self, q, k):
        with tf.name_scope('dot_attention'):
            s = (self.attention_units // self.num_heads) ** -0.5
            e = tf.einsum('abik,abjk->abij', q, k)  # <=> tf.matmul(x, y, transpose_b=True)
            e = e * s   # scale dot(q, k) with sqrt(dimension_k)
            w = self.probability_fn(e, None)  # [batch, num_heads, len_q, len_k]
        return w

    def mlp_attention(self, q, k):
        with tf.name_scope('mlp_attention'):
            dense = layers.Dense(1, 'tanh', use_bias=False, name='mlp_dense')
            q = tf.expand_dims(q, axis=3)    # [batch, num_heads, len_q, 1, dim]
            k = tf.expand_dims(k, axis=2)    # [batch, num_heads, 1, len_k, dim]
            e = dense(tf.nn.tanh(k + q))     # [batch, num_heads, len_q, len_k, 1]
            w = self.probability_fn(tf.squeeze(e, axis=-1), None)
        return w


class GSTAttention(MultiHeadAttention):
    def __init__(self,
                 num_heads=4,
                 num_tokens=10,
                 gst_units=256,
                 attention_units=128,
                 attention_type='mlp',
                 activation=None,
                 trainable=True,
                 name='gst_attention'):
        super(GSTAttention, self).__init__(num_heads=num_heads,
                                           attention_units=attention_units,
                                           attention_type=attention_type,
                                           v_layer=None,
                                           name=name)
        self.num_tokens = num_tokens
        self.gst_units = gst_units
        self.trainable = trainable

    def build(self, input_shape):
        self.gst_tokens = self.add_weight(
            name='gst_tokens',
            shape=[self.num_tokens, self.gst_units // self.num_heads],
            initializer=keras.initializers.truncated_normal(stddev=0.5, dtype=tf.float32),
            trainable=self.trainable)
        self.built = True

    def call(self, query=None, atten_weights_ph=None, training=None):
        assert query is not None or atten_weights_ph is not None

        if query is None:
            batch_size = tf.shape(atten_weights_ph)[0]
            query = tf.zeros([batch_size, 128], name='zero_gst_query')
        batch_size = tf.shape(query)[0]

        def _process_dim(query):
            query = tf.expand_dims(query, axis=1)                # [N, 1, q_dim]
            token = tf.expand_dims(self.gst_tokens, 0)           # [1, num_tokens, gst_units / num_heads]
            token = tf.tanh(tf.tile(token, [batch_size, 1, 1]))  # [N, num_tokens, gst_units / num_heads]
            token = tf.tile(token, [1, 1, self.num_heads])    # [N, num_tokens, gst_units]
            return query, token

        query, gst_tokens = _process_dim(query)
        output = super().call(query, gst_tokens,
                              atten_weights_ph=atten_weights_ph,
                              training=training)
        return output   # [N, 1, gst_units]


'''
Implementation for https://arxiv.org/abs/1906.00672
Tips: The code could be directly used in place of BahdanauMonotonicAttention in Tensorflow codes. Similar to its
base class in the Tensorflow seq2seq codebase,  you may use "hard" for hard inference, or "parallel" for training or
soft inference. "recurrent" mode in BahdanauMonotonicAttention is not supported.
If you have already trained another model using BahdanauMonotonicAttention, the model could be reused, otherwise you
possibly have to tune the score_bias_init, which, similar to that in Raffel et al., 2017, is determined a priori to
suit the moving speed of the alignments, i.e. speed of speech of your training corpus in TTS cases. So
score_bias_init=3.5, is a good one for our data, but not necessarily for yours, and our experiments find that the
results are sensitive to this bias: When the parameter is deviated from the best value, by, say, a small amount of
0.5, the whole training process may fail. sigmoid_noise=2.0 is enough in our experiments, but if you found that the
resultant alignments are far from binary, adding more noise (or annealing the noise) might be useful. Other
hyperparameters in our experiments simply follow the original Tacotron2 settings, and they work.
'''
