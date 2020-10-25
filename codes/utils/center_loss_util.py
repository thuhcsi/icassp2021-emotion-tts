import tensorflow as tf


def calc_center_loss(features, centers, emo_labels, is_l1=True):
    """
    Args:
        features:   [batch_size, dim]
        centers:    [emo_num, dim]
        emo_labels: [batch_size, emo_num]
    Returns:

    """
    features_ = tf.expand_dims(features, 1)  # [batch_size, 1, dim]
    centers_ = tf.expand_dims(centers, 0)  # [1, emo_num, dim[
    diff = features_ - centers_  # [batch_size, emo_num, dim]
    dist = tf.reduce_sum(tf.square(diff), axis=-1)  # [batch_size, emo_num]
    if is_l1:
        dist = tf.sqrt(dist)
    loss = tf.reduce_mean(tf.reduce_sum(dist * emo_labels, axis=-1))
    return loss


def update_center(features, centers, emo_labels, alpha):
    """
        Args:
            features:   [batch_size, dim]
            centers:    [emo_num, dim]
            emo_labels: [batch_size, emo_num]
        Returns:
    """
    features_ = tf.expand_dims(features, 1)  # [batch_size, 1, dim]
    centers_ = tf.expand_dims(centers, 0)  # [1, emo_num, dim]
    emo_labels_ = tf.expand_dims(emo_labels, -1)  # [batch_size, emo_num, 1]
    diff = features_ - centers_  # [batch_size, emo_num, dim]
    weighted_emo_diff = diff * emo_labels_  # [batch_size, emo_num, dim]
    sum_emo_diff = tf.reduce_sum(weighted_emo_diff, axis=0)  # [emo_num, dim]
    emo_sum = tf.clip_by_value(tf.reduce_sum(emo_labels_, axis=0), 0.001, 100)  # [emo_num, 1]
    alpha = tf.clip_by_value(alpha, 0., 1.)
    c_diff = tf.math.divide(sum_emo_diff, emo_sum)  # [emo_num, dim]
    alpha_c_diff = alpha * c_diff
    update_center_op = tf.assign_add(centers, alpha_c_diff)
    return update_center_op


def test_calc_center_loss():
    with tf.Session() as sess:
        centers = tf.Variable([[1, -1], [0, 1]], trainable=False, dtype=tf.float32)
        features = tf.constant([[-1, 0], [-1, -1], [2, 0]], dtype=tf.float32)
        emo_labels = tf.constant([[0, 1.0], [0, 1.0], [0, 1.0]], dtype=tf.float32)
        sess.run(tf.global_variables_initializer())
        print(sess.run(update_center(features, centers, emo_labels, alpha=0.5)))
        print(sess.run(update_center(features, centers, emo_labels, alpha=0.5)))
        print(sess.run(update_center(features, centers, emo_labels, alpha=0.5)))
        print(sess.run(update_center(features, centers, emo_labels, alpha=0.5)))
        print(sess.run(update_center(features, centers, emo_labels, alpha=0.5)))
        print(sess.run(update_center(features, centers, emo_labels, alpha=0.5)))
        print(sess.run(update_center(features, centers, emo_labels, alpha=0.5)))
        print(sess.run(centers))
        # loss = sess.run(calc_center_loss(features, centers, emo_labels, is_l1=True))
        # print(loss)


if __name__ == '__main__':
    test_calc_center_loss()
