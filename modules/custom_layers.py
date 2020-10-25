import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.compat.v1.logging import warn
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils

# from warnings import warn
# from utils.debug import debug_print


class ConvBlock(layers.Layer):
    def __init__(self,
                 mode,
                 conv_type,
                 channels,
                 kernel_size,
                 dropout_rate=None,
                 activation='relu',
                 batch_norm=True,
                 strides=1,
                 padding='same',
                 do_mask=False,
                 name='conv_block'):
        """stack conv(c), activation(a), batch norm(b), dropout(d) layers
        # Arguments
            mode: a str with char in ['c', 'a', 'b', 'd'] denotes the order
                of calling the corresponding layer
            conv_type: '1D' '2D' '3D' for Conv1D, Conv2D, Conv3D
        """
        super(ConvBlock, self).__init__(name=name)
        conv_map = {'1D': layers.Conv1D, '2D': layers.Conv2D, '3D': layers.Conv3D}
        conv_layer = conv_map.get(conv_type)

        self.conv = conv_layer(channels, kernel_size, strides, padding=padding, name='conv')
        self.activation = layers.Activation(activation) if activation else None
        self.bn = layers.BatchNormalization(name='bn') if batch_norm else None
        self.dropout = layers.Dropout(dropout_rate, name='drop') if dropout_rate else None

        self.do_mask = do_mask
        self.block_mode = mode
        self.supports_masking = True
        self.conv.supports_masking = True

    def call(self, x, training=None, mask=None):  # 添加mask参数, 变成掩码使用层
        layer_map = {'c': self.conv, 'b': self.bn,
                     'a': self.activation, 'd': self.dropout}

        for c in self.block_mode:
            layer = layer_map.get(c)
            # assert layer is not None, 'Layer at conv block mode but not been built'
            if layer is not None:
                if c in ['b', 'd']:
                    x = layer(x, training=training)
                else:
                    x = layer(x)
            else:
                warn(RuntimeWarning(f'ConvBlock: {c} in mode str but layer not built'))

        if self.do_mask and mask:
            mask = tf.cast(mask, 'float32')
            x = x * tf.expand_dims(mask, axis=-1)  # [N, T_in, 1]
        return x


class ZoneoutLSTMCell(layers.LSTMCell):
    def __init__(self,
                 units,
                 output_rate=0.,
                 cell_rate=None,
                 name='zolstm',
                 **kwargs):
        cell_rate = output_rate if cell_rate is None else cell_rate
        assert output_rate >= 0 and output_rate <= 1
        assert cell_rate >= 0 and cell_rate <= 1

        super(ZoneoutLSTMCell, self).__init__(units, name=name, **kwargs)
        self._cell_dropout = layers.Dropout(cell_rate, name='drop_cell')
        self._output_dropout = layers.Dropout(output_rate, name='drop_output')
        self._output_rate = output_rate
        self._cell_rate = cell_rate

    def call(self, x, state, training=None):
        '''Runs vanilla LSTM Cell and applies zoneout.
        '''
        # print('DEBUG custom_layers: training', training)  # RNN()会调用cell.call 2次
        output, new_state = super().call(x, state, training)   # <=> LSTMCell.call(self, x, state)
        pre_c, pre_h = state
        new_c, new_h = new_state

        c = (1 - self._cell_rate) * self._cell_dropout(new_c - pre_c, training) + pre_c
        h = (1 - self._output_rate) * self._output_dropout(new_h - pre_h, training) + pre_h

        new_state = [c, h]
        return output, new_state

    def get_config(self):
        config = {'output_rate': self._output_rate,
                  'cell_rate': self._cell_rate}
        super_config = super(ZoneoutLSTMCell, self).get_config()
        config.update(super_config)
        return config


class AttentionRNNCell(layers.StackedRNNCells):
    def __init__(self,
                 units=[1024, 1024],
                 zone_rate=0.1,
                 name='atten_cell'):
        self.units = units
        self.zone_rate = zone_rate
        self.lstm_cells = [ZoneoutLSTMCell(k, zone_rate) for k in units]
        super(AttentionRNNCell, self).__init__(cells=self.lstm_cells, name=name)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        return super().get_initial_state(inputs, batch_size, dtype=dtype)


class Prenet(layers.Layer):
    def __init__(self,
                 units,
                 drop_rate=0.5,
                 activation='relu',
                 name='prenet'):
        super(Prenet, self).__init__(name=name)
        self.units = units
        self.drop_rate = drop_rate
        self.activation = activation
        self.dense_layers = [layers.Dense(k, activation, name='dense') for k in units]
        self.drop_layers = [layers.Dropout(drop_rate, name='drop') for k in range(len(units))]

    def call(self, x):
        for dense, drop in zip(self.dense_layers, self.drop_layers):
            x = dense(x)
            x = drop(x, training=True)
        return x


class StopProjection(layers.Layer):
    """Projection to a scalar and through a sigmoid activation
    """
    def __init__(self,
                 units=1,
                 activation='sigmoid',
                 name='stop_projection'):
        super(StopProjection, self).__init__(name=name)
        self.units = units
        self.activation = activation
        self.dense = layers.Dense(units, None, name='dense')
        self.activation_layer = layers.Activation(activation)

    def call(self, x, training=None):
        if training is None:
            training = K.learning_phase()
        x = self.dense(x)
        x = tf_utils.smart_cond(training,
                                true_fn=lambda: tf.identity(x),
                                false_fn=lambda: self.activation_layer(x))
        return x


class FrameProjection(layers.Layer):
    """Projection layer to r * num_mels dimensions or num_mels dimensions
    """
    def __init__(self,
                 units=80,
                 activation='relu',   # our mel is normalized to [0, 1]; sygst is None
                 # activation=None,  # sygst and rayhame are None
                 name='frame_projection'):
        super(FrameProjection, self).__init__(name=name)
        self.units = units
        self.activation = activation
        self.dense = layers.Dense(units, activation, name='dense')

    def call(self, x):
        x = self.dense(x)
        return x


class Postnet(layers.Layer):
    def __init__(self,
                 num_layers,
                 channels,
                 kernel_size,
                 drop_rate,
                 output_units,
                 output_activation,
                 name='postnet'):
        """
        # Arguments
            output_units: it is usually the num_mels for mel residual connection
        """
        super(Postnet, self).__init__(name=name)
        self.cnns = [ConvBlock('cabd', '1D', channels, kernel_size, drop_rate, 'tanh')
                     for i in range(num_layers - 1)]
        self.last_cnn = ConvBlock('cbd', '1D', channels, kernel_size, drop_rate, None)
        self.dim_projection = FrameProjection(output_units, output_activation, name='postnet_proj')

    def call(self, x, training=None):
        for cnn in self.cnns:
            x = cnn(x, training)
        x = self.last_cnn(x, training)
        x = self.dim_projection(x)
        return x


class HighwayNet(layers.Layer):
    def __init__(self, units, name='highway_net'):
        super(HighwayNet, self).__init__(name=name)
        self.units = units
        self.h_layer = layers.Dense(units, 'relu', name='H')
        self.t_layer = layers.Dense(units, 'sigmoid', name='T',
                                    bias_initializer=tf.constant_initializer(-1.))

        self.supports_masking = True

    def call(self, x):
        h = self.h_layer(x)
        t = self.t_layer(x)
        x = h * t + x * (1. - t)
        return x


class CBHG(layers.Layer):
    def __init__(self,
                 K,
                 conv_channels,
                 pool_size,
                 projections,
                 highway_units,
                 highway_nums,
                 rnn_units,
                 name='cbhg'):
        super(CBHG, self).__init__(name=name)
        self.K = K
        self.conv_channels = conv_channels
        self.pool_size = pool_size
        self.projections = projections
        self.highway_units = highway_units
        self.highway_nums = highway_nums
        self.rnn_units = rnn_units

        self.conv_banks = [ConvBlock('cab', '1D', conv_channels, k, name='conv_banks')
                           for k in range(1, K + 1)]
        self.pool = layers.MaxPool1D(pool_size, strides=1, padding='same', name='pool')
        acts = ['relu'] * (len(projections) - 1) + [None]
        self.conv_projections = [ConvBlock('cab', '1D', c, 3, None, a, name='conv_projs')
                                 for c, a in zip(projections, acts)]
        self.highway_nets = [HighwayNet(highway_units, name='highway')
                             for _ in range(highway_nums)]
        self.gru_layer = layers.Bidirectional(layers.GRU(rnn_units, return_sequences=True),
                                              name='bigru')

        self.supports_masking = True

    def call(self, x, training=None):
        original_x = x
        # K conv banks: concat on the last axis to stack channels from all convs
        x = tf.concat([cnn(x, training=training) for cnn in self.conv_banks], axis=-1)

        # MaxPooling
        x = self.pool(x)

        # 2-layer conv projections
        for conv_proj in self.conv_projections:
            x = conv_proj(x, training)

        # Residual connection
        x = original_x + x

        # 4-layer HighwayNet
        if x.shape[2] != self.highway_units:
            x = layers.Dense(self.highway_units, name='dim_dense')(x)
        for highway in self.highway_nets:
            x = highway(x)

        # 1-layer bidirectional GRU
        x = self.gru_layer(x, training=training)
        return x


class ReferenceEncoder(layers.Layer):
    def __init__(self,
                 channels,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 rnn_units=128,
                 output_units=128,
                 dropout_rate=0.5,
                 name='ref_encoder'):
        super(ReferenceEncoder, self).__init__(name=name)
        self.stride = strides[0]
        self.rnn_units = rnn_units
        self.output_units = output_units

        # 6-layer conv2d
        self.cnns = [ConvBlock('cabd', '2D', c, kernel_size, dropout_rate, strides=strides)
                     for c in channels]

        # 1-layer bi-gru
        single_layer = layers.GRU(rnn_units, return_state=False)
        self.rnn = layers.Bidirectional(single_layer, name='bigru')

        # 1-layer dense
        self.dense = layers.Dense(output_units, 'tanh', name='output_dense')

    def call(self, x, x_length, training=None):
        if x.shape.ndims == 3:
            x = tf.expand_dims(x, axis=-1)

        # 6-layer conv2d
        for cnn in self.cnns:
            x = cnn(x, training)
            x_length = (x_length + self.stride - 1) // self.stride

        # stack the rest mel dim to the conv channels
        x = tf.concat(tf.unstack(x, axis=2), axis=-1)  # [N, time, mel_dim * channels]

        # 1-layer bigru with mask
        rnn_mask = tf.sequence_mask(x_length, maxlen=tf.shape(x)[1])
        x._keras_mask = rnn_mask
        x = self.rnn(x, training=training)  # [N, rnn_units * 2]

        # 1-layer dense for final output
        x = self.dense(x)  # [N, output_units]
        return x
