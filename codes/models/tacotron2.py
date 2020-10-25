# import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import backend as K

# from utils.debug import debug_print
from modules import custom_layers as cl
from modules.encoders import TacotronEncoder
from modules.encoders import Tacotron2Encoder
from modules.decoders import DecoderCell, Decoder
from modules.attention import LocationSensitiveAttention
from modules.attention import StepwiseMonotonicAttention

from models.base import TacotronBase


class Tacotron2(TacotronBase):
    def __init__(self, hp, gta=False, name='Tacotron2'):
        super(keras.Model, self).__init__(name=name)
        self.hp = hp
        self.gta = gta

        if hp.encoder_type == 'taco2':    # tacotron2 encoder
            self.encoder = Tacotron2Encoder(hp)
        elif hp.encoder_type == 'taco':   # actually is cbhg encoder
            self.encoder = TacotronEncoder(hp)
        else:
            raise ValueError('encoder_type must in [taco2, taco]')

        if hp.attention_type == 'location':  # 各个模块的定义顺序最好不要改变
            self.atten_layer = LocationSensitiveAttention(
                units=hp.attention_units,
                location_filters=hp.attention_filters,
                location_kernel_size=hp.attention_kernel_size,
                synthesis_constraint=hp.synthesis_constraint,
                synthesis_win_size=hp.synthesis_win_size,
                synthesis_softmax_temp=hp.synthesis_softmax_temp
            )
        elif hp.attention_type == 'sma':
            self.atten_layer = StepwiseMonotonicAttention(
                units=hp.attention_units,
                normalize=hp.attention_sma_normalize,
                sigmoid_noise=hp.attention_sma_sigmoid_noise,
                sigmoid_noise_seed=hp.attention_sma_sigmoid_noise_seed,
                score_bias_init=hp.attention_sma_score_bias_init,
                mode=hp.attention_sma_mode
            )
        else:
            raise ValueError('attention_type must in [location, sma]')

        self.decoder_prenet = cl.Prenet(units=hp.prenet_units, drop_rate=hp.dropout_rate)
        self.atten_rnn_cell = cl.AttentionRNNCell(units=hp.attention_rnn_units,
                                                  zone_rate=hp.zoneout_rate)
        self.frame_projection = cl.FrameProjection(units=hp.outputs_per_step * hp.num_mels,
                                                   activation=hp.frame_activation)
        self.stop_projection = cl.StopProjection(units=hp.outputs_per_step)
        self.decoder_cell = DecoderCell(prenet=self.decoder_prenet,
                                        attention_rnn=self.atten_rnn_cell,
                                        attention_layer=self.atten_layer,
                                        frame_projection=self.frame_projection,
                                        stop_projection=self.stop_projection)
        self.decoder = Decoder(hp,
                               self.decoder_cell,
                               gta=gta,
                               impute_finished=hp.impute_finished,
                               maximum_steps=hp.max_iters)

        self.postnet = cl.Postnet(num_layers=hp.postnet_cnns[0],
                                  kernel_size=hp.postnet_cnns[1],
                                  channels=hp.postnet_cnns[2],
                                  drop_rate=hp.dropout_rate,
                                  output_units=hp.num_mels,
                                  output_activation=hp.frame_activation)

        self.postcbhg = None
        if hp.post_cbhg:
            self.postcbhg = cl.CBHG(K=hp.cbhg_kernels,
                                    conv_channels=hp.cbhg_conv_channels,
                                    pool_size=hp.cbhg_pool_size,
                                    projections=[hp.cbhg_projection, hp.num_mels],
                                    highway_units=hp.cbhg_highway_units,
                                    highway_nums=hp.cbhg_highway_nums,
                                    rnn_units=hp.cbhg_rnn_units,
                                    name='post_cbhg')

        super(Tacotron2, self).__init__(hp=hp,
                                        encoder=self.encoder,
                                        decoder=self.decoder,
                                        postnet=self.postnet,
                                        postcbhg=self.postcbhg,
                                        name=name)
