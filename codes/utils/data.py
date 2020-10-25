import tensorflow as tf

from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def data_process_pipeline(hp_or_obj, meta_file, process_one_line_fun=None,
                          postprocess_fun=None, max_workers=None, **kwargs):
    """This func reads meta file and runs three custom funcs to get the final data

    This func performs the following data preprocessing pipeline:
        01 read the meta file and run the meta_fun func to parse each line in meta file, the meta_fun
            must return a tuple of two elements: data sample(e.g., a wav or image) and labels dict
        02 parse the geted data sample to the feature_fun func to get a feature data
        03 the list of feature samples is passed to the postprocess_fun to do some post preproceesing
            e.g., normalizations, fixed length padding

    # Arguments
        hp: the hyper parameter object with type 'Hparams' or a other type object where all hyper parameters
            can be accessed as it's attributes
        meta_file: the meta file where each line generally contains the path of data sample and its labels
        meta_fun: this func takes a line in meta_file and the hp object as inputs and returns a pair of
            tuple of data samples(often a list arrays) and its labels dict(a list of dicts). The signature
            of meta_fun is: def fun_name(hp, line), and its return values are:([data samples], [labels dicts])
            or (None, None) if no sample returned. Note, even there is only one sample returned, it also must
            be a list of length 1. A labels dict example is {'L1': label_1, 'L2': label_2}.
            Note: this can return (None, None) for the reason of some samples may be filtered out, e.g.,  its
                length does not meet the requirements
        feature_fun: this func takes a single data sample returned by meta_fun and the hp object as inputs,
            and returns a single feature sample. The signature is: def feature_fun(hp, sample)
        postprocess_fun: this func takes the list of all feature samples and hp as inputs and return a same
            list of postprocessed feature samples with the same length. Generally, the feature normalizations,
            fixed length padding and sorting the samples by length  are performed in this func. The signature
            is: def postprocess_fun(hp, features)
        kwargs: some extra key word arguments will be passed to all three funcs.

    # Returns
        A tuple of length 2, the first element is the list of all postprocessed features, and the second element
            is the list of all labels(each label is a list converted by the labels dict). For example, the return
            value can be: ([sample1, .., samplen], [[label1_1, label2_2], [label2_1, label_22], .., [labeln_1, labeln_2]])

    # Exceptions
        TypeError: if both value of meta_fun's return is not a list
    """
    if process_one_line_fun is None and not hasattr(hp_or_obj, 'process_one_line'):
        raise ValueError('hp_or_obj without process_one_line method and process_one_line_fun is None')
    if postprocess_fun is None and not hasattr(hp_or_obj, 'postprocess'):
        raise ValueError('hp_or_obj without postprocess method and postprocess_fun is None')

    with open(meta_file) as fr:
        lines = [line for line in fr if line.strip() and line[0] != '#']

    if hasattr(hp_or_obj, 'process_one_line'):
        process_one_line_fun = type(hp_or_obj).process_one_line
    if hasattr(hp_or_obj, 'postprocess'):
        postprocess_fun = type(hp_or_obj).postprocess

    # 处理meta line, 获取sample和标签
    print('    step 1: parsing meta and get features ...')
    num = len(lines)
    hps = [hp_or_obj] * num
    features, labels = [], []
    with ProcessPoolExecutor(max_workers) as p:
        for r in tqdm(p.map(partial(process_one_line_fun, **kwargs), hps, lines), total=num):
            ds, ls = r
            if (ds, ls) != (None, None):
                if type(ds) != list or type(ls) != list:
                    raise TypeError('meta_fun func must return a tuple of "list", not {} or {}'.format(type(ds), type(ls)))
                features += ds
                labels += ls
    '''
    for line in tqdm(lines):
        ds, ls = process_one_line_fun(hp_or_obj, line)
        if (ds, ls) != (None, None):
            if type(ds) != list or type(ls) != list:
                raise TypeError('meta_fun func must return a tuple of "list", not {} or {}'.format(type(ds), type(ls)))
            features += ds
            labels += ls
    '''

    # np.save('recola_wav01_mel_nonorm_before_post.npy', features[0])
    # print('DEBUG before post', features[0].shape, labels[0])
    # 后处理
    print('    step 2: postprocessing for features and labels ...')
    features, labels = postprocess_fun(hp_or_obj, features, labels, **kwargs)
    return features, labels


def get_class_weights(class_nums, type=0, power=1):
    if type == 0:
        return [1.] * len(class_nums)
    # 根据power, 重新计算class_nums和total
    total, class_ws = 0, class_nums.copy()
    for cls in range(len(class_ws)):
        class_ws[cls] = class_ws[cls] ** power
        total += class_ws[cls]
    # 权值取倒数后除以所有权值的均值(结果为1均值), 参考老代的做法
    if type == 1:
        wsum = 0
        for cls in class_ws:
            class_ws[cls] = 1 / class_ws[cls]
            wsum += class_ws[cls]
        wmean = wsum / len(class_nums)
        class_ws = [w / wmean for cls, w in class_ws]
    # 取倒数后乘total/2, https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    elif type == 2:
        class_ws = [0.5 * total / w for w in class_ws]
    return class_ws
