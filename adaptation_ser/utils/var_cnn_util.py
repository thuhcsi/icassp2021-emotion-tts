import tensorflow as tf


def get_mask_3d(seq_lens, max_len, dtype=tf.float32):
    """Mask for CNN hiddens. [Batch_size, max_time, channel]"""
    mask = tf.cast(tf.sequence_mask(seq_lens, max_len), dtype=dtype)
    mask = tf.expand_dims(mask, -1)
    return mask


def get_mask_4d(seq_lens, max_len, dtype=tf.float32):
    """Mask for CNN hidden. [batch_size, max_time, dim(freq), channel]"""
    mask = tf.cast(tf.sequence_mask(seq_lens, max_len), dtype=dtype)
    mask = tf.expand_dims(tf.expand_dims(mask, -1), -1)
    return mask


def _conv1d_with_seq_len(inputs,
                         filters,
                         kernel_size,
                         seq_length,
                         strides=1,
                         padding='valid',
                         dilation_rate=1,
                         use_bias=True,
                         kernel_initializer=None,
                         bias_initializer=tf.zeros_initializer(),
                         kernel_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         kernel_constraint=None,
                         bias_constraint=None,
                         trainable=True,
                         name=None,
                         reuse=None):
    if padding.lower() == 'valid':
        k = (kernel_size - 1) * dilation_rate + 1
        seq_length = seq_length - k + 1
    new_seq_len = 1 + tf.floordiv((seq_length - 1), strides)
    outputs = tf.layers.conv1d(inputs=inputs,
                               filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               data_format='channels_last',
                               dilation_rate=dilation_rate,
                               activation=None,
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer,
                               activity_regularizer=activity_regularizer,
                               kernel_constraint=kernel_constraint,
                               bias_constraint=bias_constraint,
                               trainable=trainable,
                               name=name,
                               reuse=reuse)
    return outputs, new_seq_len


def var_conv1d(inputs,
               filters,
               kernel_size,
               seq_length,
               is_seq_mask=True,
               is_bn=False,
               is_training=True,
               strides=1,
               padding='valid',
               dilation_rate=1,
               activation_fn=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               reuse=None):
    """
    1D convolution for variable length sequence input.
    :param inputs: Tensor input, [Batch_size, time_steps_ceil, channel].
    :param filters: Integer, the number of filters in the convolution.
    :param kernel_size: An integer, specifying the length of the 1D convolution window.
    :param seq_length: Tensor, [Batch_size]. Valid length of each sample.
    :param is_seq_mask: is mask position outside valid length with 0 for outputs.
    :param is_bn: batch normalization.
    :param is_training: whether is training or infering.
    :param strides: An integer, specifying the stride length of the convolution. Specifying any stride value != 1 is
                    incompatible with specifying any dilation_rate value != 1
    :param padding: One of "valid" or "same" (case-insensitive)
    :param dilation_rate: An integer, specifying the dilation rate to use for dilated convolution. Currently, specifying
                        any dilation_rate value != 1 is incompatible with specifying any strides value != 1.
    :param activation_fn: Activation function.
    :param use_bias: Boolean, whether the layer uses a bias.
    :param kernel_initializer: An initializer for the convolution kernel.
    :param bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
    :param kernel_regularizer: Optional regularizer for the convolution kernel.
    :param bias_regularizer: Optional regularizer for the bias vector.
    :param activity_regularizer: Optional regularizer function for the output.
    :param kernel_constraint: Optional projection function to be applied to the kernel after being updated by
                            an Optimizer (e.g. used to implement norm constraints or value constraints for layer
                            weights). The function must take as input the unprojected variable and must return the
                            projected variable (which must have the same shape). Constraints are not safe to use when
                            doing asynchronous distributed training.
    :param bias_constraint: Optional projection function to be applied to the bias after being updated by an Optimizer.
    :param trainable: Boolean, if True also add variables to the graph collection GraphKeys.
    :param name: A string, the name of the layer.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return:
        outputs: [Batch_size, time_steps_ceil_out, channel_out]
        seq_len: new valid time steps
    """
    if is_bn:
        use_bias = False
    outputs, seq_len = _conv1d_with_seq_len(inputs=inputs,
                                            filters=filters,
                                            kernel_size=kernel_size,
                                            seq_length=seq_length,
                                            strides=strides,
                                            padding=padding,
                                            dilation_rate=dilation_rate,
                                            use_bias=use_bias,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            bias_constraint=bias_constraint,
                                            trainable=trainable,
                                            name=name,
                                            reuse=reuse)
    if is_bn:
        outputs = tf.contrib.layers.batch_norm(inputs=outputs,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               fused=True,
                                               reuse=reuse)
    if is_seq_mask:
        mask = get_mask_3d(seq_len, tf.shape(outputs)[1], outputs.dtype)
        outputs = outputs * mask
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs, seq_len


def _conv2d_with_seq_len(inputs,
                         filters,
                         kernel_size,
                         seq_length,
                         strides,
                         padding,
                         dilation_rate,
                         use_bias=True,
                         kernel_initializer=None,
                         bias_initializer=tf.zeros_initializer(),
                         kernel_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         kernel_constraint=None,
                         bias_constraint=None,
                         trainable=True,
                         name=None,
                         reuse=None):
    """inputs, 4d tensor [batch_size, max_time, dim(freq), channel]"""
    if padding.lower() == 'valid':
        k = (kernel_size[0] - 1) * dilation_rate[0] + 1
        seq_length = seq_length - k + 1
    new_seq_len = 1 + tf.floor_div((seq_length - 1), strides[0])

    outputs = tf.layers.conv2d(inputs=inputs,
                               filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               data_format='channels_last',
                               dilation_rate=dilation_rate,
                               activation=None,
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer,
                               activity_regularizer=activity_regularizer,
                               kernel_constraint=kernel_constraint,
                               bias_constraint=bias_constraint,
                               trainable=trainable,
                               name=name,
                               reuse=reuse)
    # # todo: debug
    # print(tf.shape(inputs))
    # print('padding', padding)
    # print('dilation_rate', dilation_rate)
    # print('strides', strides)
    # print('kernel_size', kernel_size)
    # print('outputs', outputs)
    return outputs, new_seq_len


def var_conv2d(inputs,
               filters,
               kernel_size,
               seq_length,
               is_seq_mask=True,
               is_bn=False,
               is_training=True,
               strides=(1, 1),
               padding='valid',
               dilation_rate=(1, 1),
               activation_fn=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               reuse=None):
    """
        2D convolution for variable length sequence input.
        :param inputs: Tensor input, [Batch_size, time_steps_ceil, dim(freq), channel].
        :param filters: Integer, the number of filters in the convolution.
        :param kernel_size:  tuple/list of 2 integers, specifying the length of the 2D convolution window.
        :param seq_length: Tensor, [Batch_size]. Valid length of each sample.
        :param is_seq_mask: is mask position outside valid valid length with 0.
        :param is_bn: batch normalization.
        :param is_training: whether is training or infering.
        :param strides: tuple/list of 2 integers,, specifying the stride length of the convolution.
                        Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1
        :param padding: One of "valid" or "same" (case-insensitive)
        :param dilation_rate:  tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution.
                            Currently, specifying any dilation_rate value != 1 is incompatible with specifying any
                            strides value != 1.
        :param activation_fn: Activation function.
        :param use_bias: Boolean, whether the layer uses a bias.
        :param kernel_initializer: An initializer for the convolution kernel.
        :param bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
        :param kernel_regularizer: Optional regularizer for the convolution kernel.
        :param bias_regularizer: Optional regularizer for the bias vector.
        :param activity_regularizer: Optional regularizer function for the output.
        :param kernel_constraint: Optional projection function to be applied to the kernel after being updated by
                                an Optimizer (e.g. used to implement norm constraints or value constraints for layer
                                weights). The function must take as input the unprojected variable and must return the
                                projected variable (which must have the same shape). Constraints are not safe to use when
                                doing asynchronous distributed training.
        :param bias_constraint: Optional projection function to be applied to the bias after being updated by an Optimizer.
        :param trainable: Boolean, if True also add variables to the graph collection GraphKeys.
        :param name: A string, the name of the layer.
        :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
        :return:
            outputs: [Batch_size, time_steps_ceil_out, dim_out channel_out]
            seq_len: new valid time steps
        """
    if is_bn:
        use_bias = False
    outputs, seq_len = _conv2d_with_seq_len(inputs=inputs,
                                            filters=filters,
                                            kernel_size=kernel_size,
                                            seq_length=seq_length,
                                            strides=strides,
                                            padding=padding,
                                            dilation_rate=dilation_rate,
                                            use_bias=use_bias,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            bias_constraint=bias_constraint,
                                            trainable=trainable,
                                            name=name,
                                            reuse=reuse)
    if is_bn:
        outputs = tf.contrib.layers.batch_norm(inputs=outputs,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               fused=True,
                                               reuse=reuse)
    if is_seq_mask:
        mask = get_mask_4d(seq_len, tf.shape(outputs)[1], outputs.dtype)
        outputs = outputs * mask
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs, seq_len


def var_max_pooling1d(inputs,
                      pool_size,
                      strides,
                      seq_length,
                      padding='valid',
                      is_seq_mask=False,
                      name=None):
    """
    1D max pooling for variable length sequence input.
    :param inputs: The tensor over which to pool. Must have rank 3, [Batch_size, time_steps_ceil, channel].
    :param pool_size: An integer, representing the size of the pooling window.
    :param strides: An integer, specifying the strides of the pooling operation.
    :param seq_length: Tensor, [Batch_size]. Valid length of each sample.
    :param padding: A string. The padding method, either 'valid' or 'same'. Case-insensitive.
    :param is_seq_mask: is mask position outside valid length with 0 for outputs.
    :param name: A string, the name of the layer.
    :return:
        outputs: [Batch_size, time_steps_ceil_out, channel_out]
        seq_len: new valid time steps
    """
    h = tf.layers.max_pooling1d(inputs=inputs,
                                pool_size=pool_size,
                                strides=strides,
                                padding=padding,
                                name=name)
    if padding.lower() == 'valid':
        seq_length = seq_length - pool_size + 1
    new_seq_len = 1 + tf.floordiv((seq_length - 1), strides)
    if is_seq_mask:
        mask = get_mask_3d(new_seq_len, tf.shape(h)[1], h.dtype)
        h = h * mask
    return h, new_seq_len


def var_max_pooling2d(inputs,
                      pool_size,
                      strides,
                      seq_length,
                      padding='valid',
                      is_seq_mask=False,
                      name=None):
    """
    2D max pooling for variable length sequence input.
    :param inputs: The tensor over which to pool. Must have rank 4, [Batch_size, time_steps_ceil, dim(freq), channel].
    :param pool_size: tuple/list of 2 integers: (pool_height, pool_width) specifying the size of the pooling window.
    :param strides: tuple/list of 2 integers, specifying the strides of the pooling operation.
    :param seq_length: Tensor, [Batch_size]. Valid length of each sample.
    :param padding: A string. The padding method, either 'valid' or 'same'. Case-insensitive.
    :param is_seq_mask: is mask position outside valid length with 0 for outputs.
    :param name: A string, the name of the layer.
    :return:
        outputs: [Batch_size, time_steps_ceil_out, dim_out channel_out]
        seq_len: new valid time steps
    """
    h = tf.layers.max_pooling2d(inputs=inputs,
                                pool_size=pool_size,
                                strides=strides,
                                padding=padding,
                                name=name)
    if padding.lower() == 'valid':
        seq_length = seq_length - pool_size[0] + 1
    new_seq_len = 1 + tf.floordiv((seq_length - 1), strides[0])
    if is_seq_mask:
        mask = get_mask_4d(new_seq_len, tf.shape(h)[1], h.dtype)
        h = h * mask
    return h, new_seq_len


def var_conv2d_freq(inputs,
                    filters,
                    kernel_size,
                    seq_length,
                    is_seq_mask=True,
                    is_bn=False,
                    is_training=True,
                    strides=(1, 1),
                    padding='valid',
                    dilation_rate=(1, 1),
                    activation_fn=None,
                    use_bias=True,
                    kernel_initializer=None,
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    trainable=True,
                    name=None,
                    reuse=None):
    """
        2D convolution for variable length sequence input.
        :param inputs: Tensor input, [Batch_size, time_steps_ceil, dim(freq), channel].
        :param filters: Integer, the number of filters in the convolution.
        :param kernel_size:  tuple/list of 2 integers, specifying the length of the 2D convolution window.
        :param seq_length: Tensor, [Batch_size]. Valid length of each sample.
        :param is_seq_mask: is mask position outside valid valid length with 0.
        :param is_bn: batch normalization.
        :param is_training: whether is training or infering.
        :param strides: tuple/list of 2 integers,, specifying the stride length of the convolution.
                        Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1
        :param padding: One of "valid" or "same" (case-insensitive)
        :param dilation_rate:  tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution.
                            Currently, specifying any dilation_rate value != 1 is incompatible with specifying any
                            strides value != 1.
        :param activation_fn: Activation function.
        :param use_bias: Boolean, whether the layer uses a bias.
        :param kernel_initializer: An initializer for the convolution kernel.
        :param bias_initializer: An initializer for the bias vector. If None, the default initializer will be used.
        :param kernel_regularizer: Optional regularizer for the convolution kernel.
        :param bias_regularizer: Optional regularizer for the bias vector.
        :param activity_regularizer: Optional regularizer function for the output.
        :param kernel_constraint: Optional projection function to be applied to the kernel after being updated by
                                an Optimizer (e.g. used to implement norm constraints or value constraints for layer
                                weights). The function must take as input the unprojected variable and must return the
                                projected variable (which must have the same shape). Constraints are not safe to use when
                                doing asynchronous distributed training.
        :param bias_constraint: Optional projection function to be applied to the bias after being updated by an Optimizer.
        :param trainable: Boolean, if True also add variables to the graph collection GraphKeys.
        :param name: A string, the name of the layer.
        :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
        :return:
            outputs: [Batch_size, time_steps_ceil_out, dim_out channel_out]
            seq_len: new valid time steps
        """
    if is_bn:
        use_bias = False
    outputs, seq_len = _conv2d_with_seq_len(inputs=inputs,
                                            filters=filters,
                                            kernel_size=kernel_size,
                                            seq_length=seq_length,
                                            strides=strides,
                                            padding=padding,
                                            dilation_rate=dilation_rate,
                                            use_bias=use_bias,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            bias_constraint=bias_constraint,
                                            trainable=trainable,
                                            name=name,
                                            reuse=reuse)
    if is_bn:
        outputs_shape = tf.shape(outputs)
        outputs = tf.reshape(outputs,
                             [tf.shape(outputs)[0], -1, outputs.shape[2] * outputs.shape[3]])
        # outputs_shape = tf.shape(outputs)
        # outputs = tf.reshape(outputs, [tf.shape(outputs)[0], -1,
        #                                outputs_shape[2] * outputs_shape[3]])
        outputs = tf.contrib.layers.batch_norm(inputs=outputs,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               fused=True,
                                               reuse=reuse)
        outputs = tf.reshape(outputs, outputs_shape)
    if is_seq_mask:
        mask = get_mask_4d(seq_len, tf.shape(outputs)[1], outputs.dtype)
        outputs = outputs * mask
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs, seq_len
