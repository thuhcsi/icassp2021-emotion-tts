import tensorflow as tf


def trim_inputs(r, mel_inputs, spec_inputs, spec_lengths):
    """Trim the inputs' lengths to maximum multiplies of r
    """
    r = tf.cast(r, tf.int32)
    mel_inputs = tf.cast(mel_inputs, tf.float32)    # tf.cast 如果tensor dtype一样, 则返回原tensor
    spec_inputs = tf.cast(spec_inputs, tf.float32)  # 如果是np.ndarray, 则创建tensor, 且能做类型转换
    spec_lengths = tf.cast(spec_lengths, tf.int32)  # tf.convert_to_tensor, 类型不一致会报错

    max_len = tf.reduce_max(spec_lengths)
    max_len = tf.cast(max_len / r, dtype=tf.int32) * r  # cast to int32 <=> floor

    mel_inputs = mel_inputs[:, : max_len, :]
    spec_inputs = spec_inputs[:, : max_len, :]
    spec_lengths = tf.clip_by_value(spec_lengths, 0, max_len)
    return mel_inputs, spec_inputs, spec_lengths
