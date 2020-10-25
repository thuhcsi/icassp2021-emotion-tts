import tensorflow as tf


def debug_print(*args, **kwargs):
    print_op = tf.print(*args, **kwargs)
    tf.add_to_collection('print_ops', print_op)


def get_ops():
    return tf.get_collection('print_ops')
