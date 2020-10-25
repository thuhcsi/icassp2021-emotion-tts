import tensorflow as tf


def get_mel_loss(targets, outputs, spec_len=None, method='mse'):
    assert method in ['mse', 'mae'], 'loss method:{method} is not valid'

    norm_func = tf.square if method == 'mse' else tf.abs
    norm = norm_func(targets - outputs)

    if spec_len is not None:    # mask loss
        time_step = tf.shape(outputs)[1]
        mask = tf.cast(tf.sequence_mask(spec_len, time_step), tf.float32)
        sum_n = tf.reduce_sum(mask) * tf.cast(tf.shape(outputs)[-1], tf.float32)
        loss = norm * tf.expand_dims(mask, axis=-1)
    else:
        sum_n = tf.cast(tf.reduce_prod(tf.shape(outputs)), tf.float32)
        loss = norm

    name = 'mel_{}{}_loss'.format('' if spec_len is None else 'mask_', method)
    loss = tf.truediv(tf.reduce_sum(loss), sum_n, name=name)
    return loss


def get_spec_loss(targets, outputs, priority_freq_n, spec_len=None, method='mae'):
    assert method in ['mse', 'mae'], 'loss method:{method} is not valid'

    norm_func = tf.square if method == 'mse' else tf.abs
    norm = norm_func(targets - outputs)
    priority_freq_n = tf.cast(priority_freq_n, tf.int32)

    if spec_len is not None:    # mask loss
        time_step = tf.shape(outputs)[1]
        mask = tf.cast(tf.sequence_mask(spec_len, time_step), tf.float32)
        sum_n = tf.reduce_sum(mask) * tf.cast(tf.shape(outputs)[-1], tf.float32)
        sum_m = tf.reduce_sum(mask) * tf.cast(priority_freq_n, tf.float32)
        loss = norm * tf.expand_dims(mask, axis=-1)
    else:
        sum_n = tf.cast(tf.reduce_prod(tf.shape(outputs)), tf.float32)
        sum_m = tf.cast(tf.reduce_prod(tf.shape(outputs)[: 2]) * priority_freq_n, tf.float32)
        loss = norm

    name = 'spec_{}{}_loss'.format('' if spec_len is None else 'mask_', method)
    # Prioritize loss for frequencies under 2000 Hz.
    loss_low = loss[:, :, 0: priority_freq_n]
    loss = [tf.reduce_sum(loss) / sum_n, tf.reduce_sum(loss_low) / sum_m]
    loss = tf.tensordot(loss, [0.5, 0.5], axes=1, name=name)
    return loss


def get_stop_loss(targets, outputs, outputs_per_step=None, spec_len=None, do_mask=False, pos_weight=1.):
    time_step = tf.shape(outputs)[1]
    if targets is None:
        assert spec_len is not None, 'stop token targets and spec_len can not be both None'
        pre_zero_len = spec_len - outputs_per_step
        pre_zero_mask = tf.cast(tf.sequence_mask(pre_zero_len, time_step), tf.float32)
        targets = tf.ones_like(outputs) - pre_zero_mask

    loss = tf.nn.weighted_cross_entropy_with_logits(labels=targets, logits=outputs, pos_weight=pos_weight)

    if do_mask:
        assert spec_len is not None, 'do_mask=True requires spec_len is not None'
        mask = tf.cast(tf.sequence_mask(spec_len, time_step), tf.float32)
        sum_n = tf.reduce_sum(mask)
        loss = loss * mask
    else:
        sum_n = tf.cast(tf.reduce_prod(tf.shape(outputs)), tf.float32)

    name = 'stop_{}loss'.format('mask_' if do_mask else '')
    loss = tf.truediv(tf.reduce_sum(loss), sum_n, name=name)
    return loss
