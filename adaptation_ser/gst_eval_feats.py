import tensorflow as tf
import numpy as np
from cross_lingual import d_set, eval_util, gst_model
from utils import parser_util, cfg_process
import os


class ModelEvalFeats(object):

    def __init__(self, hp):
        self.hp = hp
        self.sess = tf.Session()
        model_func = gst_model.CRModel

        eval_ds = d_set.DataSet(is_repeat=False, hp=hp,
                                tfr_path=self.hp.eval_tfr)
        eval_iter = eval_ds.get_iter()
        self.eval_iter_initializer = eval_iter.initializer
        features = eval_iter.get_next()
        if self.hp.label_type == 'arousal':
            labels = features['arousal']
        else:
            labels = features['valance']
        # todo, connect emo label to sentence name(uid)
        self.uid_iter_item = features['uid']
        self.eval_model = model_func(self.hp, features['spec'],
                                     features['seq_len'], labels)
        # self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, self.hp.restore_ckpt)

    def evaluate(self):
        feats = []
        MAX_LOOP = 9999
        self.sess.run(self.eval_iter_initializer)
        for _ in range(MAX_LOOP):
            try:
                batched_feats= self.sess.run(
                    self.eval_model.feats,
                    feed_dict={
                        self.eval_model.fc_kprob_ph: 1,
                        self.eval_model.is_training_ph: False
                    })
                feats.append(batched_feats)
            except tf.errors.OutOfRangeError:
                break
        feats_np = np.concatenate(feats, axis=0)
        os.makedirs(self.hp.feat_npy_fold, exist_ok=True)
        feat_npy_path = os.path.join(self.hp.feat_npy_fold, self.hp.feat_npy_name)
        np.save(feat_npy_path, feats_np)
        return feats_np

    def run(self):
        self.evaluate()
        self.sess.close()


def add_arguments(parser):
    """Build ArgumentParser"""
    parser.add_argument('--config_file', type=str,
                        default='./gst_cfg/recola2iemocap.yml',
                        help='config file about hparams')
    parser.add_argument('--config_name', type=str, default='default',
                        help='config name for hparams')


def main():
    parser = parser_util.MyArgumentParser()
    add_arguments(parser)
    argc, flags_dict = parser.parse_to_dict()
    yparams = cfg_process.YParams(argc.config_file, argc.config_name)
    yparams = cfg_process.HpsGSTPreprocessor(yparams, flags_dict).preprocess()
    yparams.save()
    model_eval = ModelEvalFeats(yparams)
    model_eval.run()


if __name__ == '__main__':
    main()
