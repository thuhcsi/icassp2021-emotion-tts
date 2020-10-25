# import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers


# from taco2_hparams import hp
from modules import custom_layers as cs


class Tacotron2Encoder(layers.Layer):
    def __init__(self, hp, name='taco2_encoder'):
        super(Tacotron2Encoder, self).__init__(name=name)
        # embedding layer
        self.embed_layer = layers.Embedding(hp.num_symbols, hp.embedding_dim, mask_zero=True)

        # 3-layer conv1d
        cnns_num, ksize, channels = hp.encoder_cnns
        self.cnns = [cs.ConvBlock('cabd', '1D', channels, ksize, hp.dropout_rate)
                     for i in range(cnns_num)]

        # 1-layer bi-lstm
        units, zo_rate = hp.encoder_rnns_units, hp.zoneout_rate
        single_layer = layers.RNN(cs.ZoneoutLSTMCell(units, zo_rate),
                                  return_sequences=True)
        # with mask, outputs zero for time step that mask is 0
        self.rnn = layers.Bidirectional(single_layer, name='bilstm')

    def call(self, x, training=None):
        x = self.embed_output = self.embed_layer(x)   # x是mask_tensor, 即, x._keras_tensor != None
        for cnn in self.cnns:
            x = cnn(x, training=training)             # 有bn和dropout, 必须传递training
        x = self.rnn(x, training=training)  # bi-rnn必须用keyword arguments传递
        return x


class TacotronEncoder(layers.Layer):
    def __init__(self, hp, name='taco_encoder'):
        super(TacotronEncoder, self).__init__(name=name)

        # embedding layer
        self.embed_layer = layers.Embedding(hp.num_symbols, hp.embedding_dim, mask_zero=True)

        # 2-dense-layer prenet
        # self.prenet = cs.Prenet(units=[256, 128], name='prenet')
        self.prenet = cs.Prenet(units=[256, 256], name='prenet')

        # cbhg block
        """
        self.cbhg = cs.CBHG(K=16,
                            conv_channels=128,
                            pool_size=2,
                            projections=[128, 128],
                            highway_units=128,
                            highway_nums=4,
                            rnn_units=128,  # 标准tacotron是128
                            name='cbhg')
        """
        self.cbhg = cs.CBHG(K=16,
                            conv_channels=256,
                            pool_size=2,
                            projections=[256, 256],
                            highway_units=256,
                            highway_nums=4,
                            rnn_units=256,  # 标准tacotron是128
                            name='cbhg')

    def call(self, x, training=None):
        x = self.embed_output = self.embed_layer(x)
        x = self.prenet(x)
        x = self.cbhg(x, training=training)
        return x
