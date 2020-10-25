import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.python import rnn
from tensorflow.python.util import nest
from tensorflow.python.keras.utils import tf_utils

from collections import namedtuple

# from utils.debug import debug_print

"""
# 此种写法不支持tf.convert_to_tensor()
class DecoderCellState:
    def __init__(self,
                 attention_rnn_state,
                 attention_context,
                 attention_alignments,
                 attention_accum_alignments,
                 decode_rnn_state):
        self.attention_rnn_state = attention_rnn_state
        self.attention_context = attention_context
        self.attention_alignments = attention_alignments
        self.attention_accum_alignments = attention_accum_alignments
        self.decode_rnn_state = decode_rnn_state
"""

DecoderCellState = namedtuple('DecoderCellState',
                              ['attention_state',
                               'attention_rnn_state',
                               'decode_rnn_state'])


class DecoderCell(layers.Layer):
    def __init__(self,
                 prenet,
                 attention_rnn,
                 attention_layer,
                 frame_projection,
                 stop_projection,
                 decode_rnn=None,
                 text_input_shape=None,
                 name='decoder_cell'):
        """
        # Arguments
            text_input_shape: the shape of encoder outputs, must be get by tf.shape
        """
        super(DecoderCell, self).__init__(name=name)
        self.prenet = prenet
        self.attention_rnn = attention_rnn
        self.attention_layer = attention_layer
        self.frame_projection = frame_projection
        self.stop_projection = stop_projection
        self.decode_rnn = decode_rnn

        self._batch_size = None
        self._text_time_steps = None

    def call(self, x, state, training=None):
        # prenet
        prenet_output = self.prenet(x)
        atten_rnn_input = tf.concat([prenet_output,
                                     state.attention_state.context],
                                    axis=-1, name='prenet_concat')
        # attention rnn
        atten_rnn_output, atten_rnn_state = self.attention_rnn(atten_rnn_input,
                                                               state.attention_rnn_state,
                                                               training=training)
        # nvidia-torch -> drop(atten_rnn_output, 0.1)

        # attention computation
        atten_context, atten_state = self.attention_layer(atten_rnn_output,
                                                          state.attention_state,
                                                          training=training)
        atten_rnn_context_cat = tf.concat([atten_rnn_output, atten_context],
                                          axis=-1, name='atten_rnn_concat')
        if self.decode_rnn:
            decode_rnn_output, decode_rnn_state = self.decode_rnn(atten_rnn_context_cat,
                                                                  state.decode_rnn_state,
                                                                  training=training)
            # nvidia-torch -> drop(decode_rnn_output, 0.1)
            projection_input = tf.concat([decode_rnn_output, atten_context],
                                         axis=-1, name='decode_rnn_concat')
        else:
            decode_rnn_state = tf.zeros(tf.shape(x)[0])
            projection_input = atten_rnn_context_cat

        cell_outputs = self.frame_projection(projection_input)
        stop_tokens = self.stop_projection(projection_input)

        next_state = DecoderCellState(atten_state,
                                      atten_rnn_state,
                                      decode_rnn_state)
        return (cell_outputs, stop_tokens, atten_state.alignments), next_state

    def set_batch_timesteps(self, batch_size, text_time_steps):
        self.batch_size = batch_size
        self.text_time_steps = text_time_steps

    def get_init_state(self):
        assert self.batch_size is not None, 'batch_size is None, can not get initial state'

        attention_state = self.attention_layer.init_state
        attention_rnn_state = self.attention_rnn.get_initial_state(batch_size=self.batch_size)
        if self.decode_rnn:
            decode_rnn_state = self.decode_rnn.get_initial_state(batch_size=self.batch_size)
        else:
            decode_rnn_state = tf.zeros(self.batch_size)  # 不能写成None
        return DecoderCellState(attention_state,
                                attention_rnn_state,
                                decode_rnn_state)

    @property
    def output_size(self):
        return (self.frame_projection.units,
                self.stop_projection.units,
                self.text_time_steps)

    @property
    def output_dtype(self):
        return (tf.float32, tf.float32, tf.float32)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        assert isinstance(batch_size, (tf.Tensor, int))
        self._batch_size = batch_size

    @property
    def text_time_steps(self):
        return self._text_time_steps

    @text_time_steps.setter
    def text_time_steps(self, text_time_steps):
        assert isinstance(text_time_steps, tf.Tensor)
        self._text_time_steps = text_time_steps


class Decoder(layers.Layer):
    def __init__(self,
                 hp,
                 decoder_cell,
                 gta=False,     # ground truth alignment
                 impute_finished=False,
                 maximum_steps=2000,
                 parallel_iterations=32,
                 swap_memory=False,
                 name='decoder'):
        super(Decoder, self).__init__(name=name)
        self.hp = hp
        self.cell = decoder_cell
        self.gta = gta
        self.impute_finished = impute_finished
        self.maximum_steps = maximum_steps
        self.parallel_iterations = parallel_iterations
        self.swap_memory = swap_memory

        self.r = hp.outputs_per_step
        self.feed_last_frame = hp.feed_last_frame
        self.input_dim = hp.num_mels * (1 if hp.feed_last_frame else self.r)

    def set_inputs(self, inputs, inputs_lengths):
        self._inputs, self._inputs_lengths = None, None
        if inputs is not None and inputs_lengths is not None:
            batch = tf.shape(inputs)[0]
            self._inputs_lengths = tf.cast(inputs_lengths / self.r, dtype=tf.int32)
            if self.feed_last_frame:
                self._inputs = inputs[:, self.r - 1::self.r, :]
            else:
                self._inputs = tf.reshape(inputs, [batch, -1, self.input_dim])

    def get_init_values(self):
        with tf.name_scope('while_loop_init_values'):
            init_time = tf.constant(0, dtype=tf.int32)
            init_state = self.cell.get_init_state()
            init_inputs = tf.zeros(shape=(self.batch_size, self.input_dim))
            init_finished = tf.tile([False], [self.batch_size])
            init_seq_lengths = tf.zeros(shape=self.batch_size, dtype=tf.int32)

            def _create_tensor_array(s, d):
                return tf.TensorArray(size=0, dtype=d,
                                      dynamic_size=True,
                                      # element_shape=(self.batch_size, s))
                                      element_shape=None)
            init_outputs_ta = nest.map_structure(_create_tensor_array,
                                                 self.cell.output_size,
                                                 self.cell.output_dtype)

            return init_time, init_outputs_ta, init_state, init_inputs, init_finished, init_seq_lengths

    def _next_inputs(self, time, outputs):
        mel_output, stop_token = outputs[:2]  # [N, mel_num * r], [N, r]

        def _true_fn():  # training=True or gta=True
            next_inputs = self._inputs[:, time, :]
            finished = (time + 1 >= self._inputs_lengths)  # 下一个时间步是否完成
            # finished = (time + 1 >= tf.shape(self._inputs)[1])  # 防止出现整个batch最大长度小于time_step
            # 此句报错(构建计算图时), 说迭代前后shape不一致, 可能是因为tf.shape(self._inputs)[1]的原因
            # 后面是直接将原始的输入做trim到r的整数倍就可以直接用time>=self._input_lengths了
            return next_inputs, finished

        def _false_fn():
            next_inputs = mel_output[:, -self.input_dim:]
            finished = tf.cast(tf.round(stop_token), tf.bool)  # >0.5->1, <=0.5->0
            finished = tf.reduce_any(finished, axis=1)   # maximum_iteraions is set at tf.while_loop
            return next_inputs, finished
        assert not (self.gta and self._inputs is None)
        pred = tf.logical_or(self.training, self.gta)
        return tf_utils.smart_cond(pred, _true_fn, _false_fn)

    def step(self, time, inputs, state):
        """
        # Arguments
            time: current time step
            inputs: current time step inputs
            state: previous time step state
        # Returns
            outputs: current time step outputs
            next_state: current time step sate
            next_inputs: next time step inputs
            next_finished: whether the next time step is finished
        # ps: when at the i-th time step
            inputs[i] = targets[i - 1]
            param state = state[i - 1]
            outputs[i] = targets[i] = inputs[i + 1]
            next_state = state[i] = param state[i + 1]
            next_finished = finished[i + 1]
        """
        outputs, next_state = self.cell(inputs, state, self.training)
        next_inputs, next_finished = self._next_inputs(time, outputs)
        return outputs, next_state, next_inputs, next_finished

    def call(self, inputs=None, inputs_lengths=None, batch_size=None, training=None):
        """
        # Arguments
            inputs: mel spectrum inputs, with shape [N, frame_nums, num_mels]
                it can be None when in inference phase
            inputs_lengths: mel spectrum lengths
            batch_size: used in inference phase where inputs is None
        """
        assert inputs is not None or batch_size is not None

        self.training = training
        self.batch_size = batch_size if inputs is None else tf.shape(inputs)[0]
        self.set_inputs(inputs, inputs_lengths)
        init_values = self.get_init_values()
        zero_outputs = nest.map_structure(lambda s, d: tf.zeros((self.batch_size, s), d),
                                          self.cell.output_size,
                                          self.cell.output_dtype)

        def condition(unused_time, unused_outputs_ta, unused_state,
                      unused_inputs, finished, unused_seq_lengths):
            return tf.logical_not(tf.reduce_all(finished))

        def body(time, outputs_ta, state, inputs, finished, seq_lengths):
            """
            # Arguments
                time: current time step
                outputs_ta: the tensor array outputs, for collecting outputs at each time step
                inputs: current time step inputs
                finished: whether current time step is marked as finished
                seq_lengths: the actually lengths for each sample in batch(only used at inference phase))

            # Returns
                final_outputs: a sequence of outputs from decoder cell, i.e., (mel, stop, alignments)
                final_state: the last state from decoder cell
                final_seq_lengths: the actually lengths of outputs sequence (used at inference)
            """
            (outputs, next_state, next_inputs, next_finished) = self.step(time, inputs, state)
            next_finished = tf.logical_or(finished, next_finished)
            next_seq_lengths = seq_lengths + tf.cast(tf.logical_not(finished), dtype=tf.int32)

            nest.assert_same_structure(state, next_state)
            nest.assert_same_structure(outputs_ta, outputs)
            nest.assert_same_structure(inputs, next_inputs)
            # note: the following two lines use the 'finished' instead of 'next_finished'
            # the output at time after finished will be zero
            if self.impute_finished:  # output zero and copy the state
                emit = nest.map_structure(lambda zero, out: tf.where(finished, zero, out),
                                          zero_outputs, outputs)
                # the state at time after finished will be copied the last state
                next_state = nest.map_structure(lambda cur, new: tf.where(finished, cur, new),
                                                state, next_state)
            else:
                emit = outputs
            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                            outputs_ta, emit)
            return (time + 1, outputs_ta, next_state, next_inputs, next_finished, next_seq_lengths)

        res = tf.while_loop(condition, body, loop_vars=init_values,
                            parallel_iterations=self.parallel_iterations,
                            maximum_iterations=self.maximum_steps,
                            swap_memory=self.swap_memory)
        final_outputs_ta = res[1]
        final_state = res[2]
        final_seq_lengths = res[5]

        final_outputs = nest.map_structure(lambda ta: ta.stack(),
                                           final_outputs_ta)
        final_outputs = nest.map_structure(rnn._transpose_batch_time,
                                           final_outputs)

        return final_outputs, final_state, final_seq_lengths
