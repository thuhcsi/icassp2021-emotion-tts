import tensorflow as tf
import math


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


class MultiheadAttention():
    '''Computes the multi-head attention as described in
    https://arxiv.org/abs/1706.03762.
    Args:
      num_heads: The number of attention heads.
      query: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
      value: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
        If ``None``, computes self-attention.
      num_units: The number of hidden units. If not set, it is set to the input
        dimension.
      attention_type: a string, either "dot_attention", "mlp_attention".
    Returns:
       The concatenated attention context of each head.
    '''

    def __init__(self,
                 query,
                 value,
                 num_heads=4,
                 attention_type='mlp_attention',
                 num_units=None,
                 normalize=True):
        self.query = query
        self.value = value
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.num_units = num_units or query.get_shape().as_list()[-1]
        self.normalize = normalize

    def multi_head_attention(self, is_weight=False):
        if self.num_units % self.num_heads != 0:
            raise ValueError("Multi head attention requires that num_units is a"
                             " multiple of {}".format(self.num_heads))

        with tf.variable_scope("Multihead-attention"):
            q = tf.layers.conv1d(self.query, self.num_units, 1)
            k = tf.layers.conv1d(self.value, self.num_units, 1)
            v = self.value
            qs, ks, vs = self._split_heads(q, k, v)
            if self.attention_type == 'mlp_attention':
                style_embeddings = self._mlp_attention(qs, ks, vs, is_weight)
            elif self.attention_type == 'dot_attention':
                style_embeddings = self._dot_product(qs, ks, vs, is_weight)
            else:
                raise ValueError(
                    'Only mlp_attention and dot_attention are supported')
            if is_weight:
                h = style_embeddings
                return tf.reshape(h, [tf.shape(h)[0],
                                      h.shape[1] * h.shape[2] * h.shape[3]])

            return tf.squeeze(self._combine_heads(style_embeddings), axis=1)

    def _split_heads(self, q, k, v):
        '''Split the channels into multiple heads

        Returns:
             Tensors with shape [batch, num_heads, length_x, dim_x/num_heads]
        '''
        qs = tf.transpose(self._split_last_dimension(q, self.num_heads),
                          [0, 2, 1, 3])
        ks = tf.transpose(self._split_last_dimension(k, self.num_heads),
                          [0, 2, 1, 3])
        v_shape = shape_list(v)
        vs = tf.tile(tf.expand_dims(v, axis=1), [1, self.num_heads, 1, 1])
        return qs, ks, vs

    def _split_last_dimension(self, x, num_heads):
        '''Reshape x to num_heads

        Returns:
            a Tensor with shape [batch, length_x, num_heads, dim_x/num_heads]
        '''
        x_shape = shape_list(x)
        dim = x_shape[-1]
        assert dim % num_heads == 0
        return tf.reshape(x, x_shape[:-1] + [num_heads, dim // num_heads])

    def _dot_product(self, qs, ks, vs, is_weight=False):
        '''dot-product computation

        Returns:
            a context vector with shape [batch, num_heads, length_q, dim_vs]
        '''
        qk = tf.matmul(qs, ks, transpose_b=True)
        scale_factor = (self.num_units // self.num_heads) ** -0.5
        if self.normalize:
            qk *= scale_factor
        weights = tf.nn.softmax(qk, name="dot_attention_weights")
        if is_weight:
            return weights
            # h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
            # return tf.reshape(weights, [tf.shape(weights)[0], weights.shape[1] * weights.shape[2] * weights.shape[3]])
        context = tf.matmul(weights, vs)
        return context

    def _mlp_attention(self, qs, ks, vs, is_weight=False):
        '''MLP computation modified from https://github.com/npuichigo

        Returns:
            a context vector with shape [batch, num_heads, length_q, dim_vs]
        '''
        num_units = qs.get_shape()[-1].value
        dtype = qs.dtype

        v = tf.get_variable("attention_v", [num_units], dtype=dtype)
        if self.normalize:
            # https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py#L470
            # Scalar used in weight normalization
            g = tf.get_variable(
                "attention_g", dtype=dtype,
                initializer=math.sqrt((1. / num_units)))
            # Bias added prior to the nonlinearity
            b = tf.get_variable(
                "attention_b", [num_units], dtype=dtype,
                initializer=tf.zeros_initializer())
            # normed_v = g * v / ||v||
            normed_v = g * v * tf.rsqrt(
                tf.reduce_sum(tf.square(v)))
            # Single layer multilayer perceptron.
            add = tf.reduce_sum(normed_v * tf.tanh(ks + qs + b), [-1],
                                keep_dims=True)
        else:
            # Single layer multilayer perceptron.
            add = tf.reduce_sum(v * tf.tanh(ks + qs), [-1], keep_dims=True)

        # Compute attention weights.
        weights = tf.nn.softmax(tf.transpose(add, [0, 1, 3, 2]),
                                name="mlp_attention_weights")
        if is_weight:
            return weights
        # Compute attention context.
        context = tf.matmul(weights, vs)
        return context

    def _combine_heads(self, x):
        '''Combine all heads

           Returns:
               a Tensor with shape [batch, length_x, shape_x[-1] * shape_x[-3]]
        '''
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        return tf.reshape(x, x_shape[:-2] + [self.num_heads * x_shape[-1]])
