import tensorflow as tf
from utils import var_cnn_util as vcu
from collections import defaultdict
from utils.multihead_attention import MultiheadAttention


class BaseModel(object):  # CRNN

    def __init__(self, hps, inputs, seq_lens, labels):
        self.hps = hps
        self.inputs = inputs
        self.seq_lens = seq_lens
        self.labels = labels
        self.fc_kprob_ph = tf.placeholder(tf.float32, shape=[],
                                          name='fc_kprob_ph')
        self.is_training_ph = tf.placeholder(tf.bool, shape=[],
                                             name='is_training_ph')
        self.lr_ph = tf.placeholder(tf.float32, shape=[], name='lr_ph')
        self.logits = None
        self.pre = None  # predict value
        self.loss = None  # loss
        # self.metric_d = None
        self.debug_d = defaultdict()
        self.train_op = None
        self.feats = None
        self.build_graph()

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    def build_graph(self):
        raise NotImplementedError('build_graph function not implements yet')


class CRModel(BaseModel):
    def cnn(self, inputs, seq_lens):

        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[7, 7], [3, 3], [3, 3], [3, 3]]
        filter_nums = [48, 64, 80, 96]
        strides = [[2, 2], [1, 1], [1, 1], [1, 1]]
        for ker_size, filter_num, stride, is_pool in zip(kernel_sizes,
                                                         filter_nums, strides,
                                                         is_poolings):
            i += 1
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = vcu.var_conv2d(inputs=h, filters=filter_num,
                                             kernel_size=ker_size,
                                             seq_length=seq_lens,
                                             strides=stride, padding='same',
                                             use_bias=True,
                                             is_seq_mask=self.hps.is_var_cnn_mask,
                                             is_bn=self.hps.is_bn,
                                             activation_fn=tf.nn.relu,
                                             is_training=self.is_training_ph)
                if is_pool:
                    h, seq_lens = vcu.var_max_pooling2d(inputs=h,
                                                        pool_size=[2, 2],
                                                        strides=[2, 2],
                                                        seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens

    def rnn(self, inputs, seq_lens):
        rnn_hidden_size = 128
        with tf.name_scope('rnn'):
            rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_size)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell,
                                                             inputs,
                                                             sequence_length=seq_lens,
                                                             dtype=tf.float32,
                                                             swap_memory=True)

            rng = tf.range(0, tf.shape(seq_lens)[0])
            indexes = tf.stack([rng, seq_lens - 1], axis=1, name="indexes")
            self.debug_d['seq_lens'] = seq_lens
            self.debug_d['indexes'] = indexes
            fw_outputs = tf.gather_nd(outputs[0], indexes)
            bw_outputs = outputs[1][:, 0]
            outputs_concat = tf.concat([fw_outputs, bw_outputs], axis=1)
        return outputs_concat

    def fc(self, inputs):
        out_dim = self.hps.out_dim
        fc_hidden = 64
        inputs = tf.nn.dropout(inputs, self.fc_kprob_ph)
        with tf.name_scope('fc1'):
            h_fc1_act = tf.layers.dense(inputs=inputs, units=fc_hidden,
                                        activation=tf.keras.layers.PReLU())
        with tf.name_scope('fc2'):
            h_fc2 = tf.layers.dense(inputs=h_fc1_act, units=out_dim,
                                    activation=None)
        h_fc = h_fc2
        hid_fc = h_fc1_act
        return h_fc, hid_fc

    def model_fn(self):
        print('CRModel')
        h_cnn, seq_lens = self.cnn(self.inputs, self.seq_lens)
        h_rnn = self.rnn(h_cnn, seq_lens)
        self.feats = h_rnn
        logits, hid_fc = self.fc(h_rnn)
        return logits

    def get_loss(self):

        positives = tf.cast(self.labels, tf.float32)
        negatives = 1 - positives
        pos_count = tf.reduce_sum(positives)
        neg_count = tf.reduce_sum(negatives)
        weights = (positives * neg_count + negatives * pos_count) * 2 / (
                pos_count + neg_count)
        # weights = positives * self.hps.positive_weight + (1.0 - positives)
        with tf.name_scope('ce_loss'):
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=self.labels,
                logits=self.logits,
                weights=weights,
                reduction=tf.losses.Reduction.MEAN
            )
        return loss

    def get_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer,
                                                                clip_norm=5.0)
        return optimizer.minimize(self.loss)

    def build_graph(self):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
            self.logits = self.model_fn()
            self.loss = self.get_loss()
            self.train_op = self.get_train_op()


class GSTModel(CRModel):

    def __init__(self, hps, inputs, seq_lens, labels):
        with tf.variable_scope('style_tokens', reuse=tf.AUTO_REUSE) as scope:
            gst_tokens = tf.get_variable(
                'style_tokens',
                [hps.num_gst, hps.style_embed_depth // hps.num_heads],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5))
            self.gst_tokens = gst_tokens
        super(GSTModel, self).__init__(hps, inputs, seq_lens, labels)

    def gst_func(self, h_rnn):
        batch_size = tf.shape(h_rnn)[0]
        style_attention = MultiheadAttention(
            tf.expand_dims(h_rnn, axis=1),  # [N, 1, 128]
            tf.tanh(tf.tile(tf.expand_dims(self.gst_tokens, axis=0),
                            [batch_size, 1, 1])),
            # [N, hp.num_gst, 256/hp.num_heads]
            num_heads=self.hps.num_heads,
            num_units=self.hps.style_att_dim,
            attention_type=self.hps.style_att_type)
        style_embeddings = style_attention.multi_head_attention()
        return style_embeddings

    def model_fn(self):
        print('GSTModel')
        h_cnn, seq_lens = self.cnn(self.inputs, self.seq_lens)
        h_rnn = self.rnn(h_cnn, seq_lens)
        style_embeddings = self.gst_func(h_rnn)
        print(tf.shape(style_embeddings))
        logits, hid_fc = self.fc(style_embeddings)
        return logits


class GST2Model(GSTModel):

    def fc(self, inputs):
        out_dim = self.hps.out_dim
        fc_hidden = 16
        inputs = tf.nn.dropout(inputs, self.fc_kprob_ph)
        with tf.name_scope('fc1'):
            h_fc1_act = tf.layers.dense(inputs=inputs, units=fc_hidden,
                                        activation=tf.keras.layers.PReLU())
        with tf.name_scope('fc2'):
            h_fc2 = tf.layers.dense(inputs=h_fc1_act, units=out_dim,
                                    activation=None)
        h_fc = h_fc2
        hid_fc = h_fc1_act
        return h_fc, hid_fc

    def gst_func(self, h_rnn):
        batch_size = tf.shape(h_rnn)[0]
        style_attention = MultiheadAttention(
            tf.expand_dims(h_rnn, axis=1),  # [N, 1, 128]
            tf.tanh(tf.tile(tf.expand_dims(self.gst_tokens, axis=0),
                            [batch_size, 1, 1])),
            # [N, hp.num_gst, 256/hp.num_heads]
            num_heads=self.hps.num_heads,
            num_units=self.hps.style_att_dim,
            attention_type=self.hps.style_att_type)
        style_embeddings = style_attention.multi_head_attention(is_weight=True)
        return style_embeddings

    def model_fn(self):
        print('GST2Model')
        h_cnn, seq_lens = self.cnn(self.inputs, self.seq_lens)
        h_rnn = self.rnn(h_cnn, seq_lens)
        style_embeddings = self.gst_func(h_rnn)
        # print(tf.shape(style_embeddings))
        logits, hid_fc = self.fc(style_embeddings)
        return logits
