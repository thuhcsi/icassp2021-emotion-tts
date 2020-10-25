import tensorflow as tf


def ce_loss(soft_labels, logits):
    probs = tf.clip_by_value(tf.nn.softmax(logits, axis=-1), 1e-10, 10)
    ce = -tf.reduce_mean(tf.reduce_sum(soft_labels * tf.log(probs), axis=-1))
    return ce
