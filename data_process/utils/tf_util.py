import tensorflow as tf


def bytes_feature(x):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))


def int64_feature(x):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))


def float_feature(x):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[x]))
