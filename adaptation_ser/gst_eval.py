import tensorflow as tf
import numpy as np
from cross_lingual import d_set, eval_util, gst_model
from utils import parser_util, cfg_process


class ModelEval(object):

    def __init__(self, hp):
        self.hp = hp
        self.sess = tf.Session()

        if self.hp.model_type == 'gst':
            model_func = gst_model.GSTModel
        elif self.hp.model_type == 'r_gst2':
            model_func = gst_model.GST2Model
        else:
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
        prs = []
        gts = []
        MAX_LOOP = 9999
        self.sess.run(self.eval_iter_initializer)
        for _ in range(MAX_LOOP):
            try:
                batched_logits, batched_gt = self.sess.run(
                    (self.eval_model.logits, self.eval_model.labels),
                    feed_dict={
                        self.eval_model.fc_kprob_ph: 1,
                        self.eval_model.is_training_ph: False
                    })
                batched_pr = np.argmax(batched_logits, axis=-1)
                prs += list(batched_pr)
                gts += list(batched_gt)
            except tf.errors.OutOfRangeError:
                break
        eval_util.get_ua_wa(prs, gts, is_print=True,
                            result_f=self.hp.result_file)

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
    model_eval = ModelEval(yparams)
    model_eval.run()


if __name__ == '__main__':
    main()
