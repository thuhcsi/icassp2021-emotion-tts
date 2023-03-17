import tensorflow as tf
from cross_lingual import d_set, eval_util, gst_model
from utils import parser_util, cfg_process
import numpy as np
import os


class ModelTrain(object):

    def __init__(self, hp):
        self.hp = hp
        self.sess = tf.Session()

        if self.hp.model_type == 'gst':
            model_func = gst_model.GSTModel
        elif self.hp.model_type == 'r_gst2':
            model_func = gst_model.GST2Model
        else:
            model_func = gst_model.CRModel

        # train model
        train_ds = d_set.DataSet(is_repeat=True, tfr_path=self.hp.train_tfr,
                                 hp=self.hp)
        train_iter = train_ds.get_iter()
        self.train_iter_initializer = train_iter.initializer
        train_features = train_iter.get_next()
        if self.hp.label_type == 'arousal':
            train_labels = train_features['arousal']
        else:
            train_labels = train_features['valance']

        self.train_model = model_func(self.hp, train_features['spec'],
                                      train_features['seq_len'],
                                      train_labels)

        # valid model
        valid_ds = d_set.DataSet(is_repeat=False, tfr_path=self.hp.valid_tfr,
                                 hp=self.hp)
        valid_iter = valid_ds.get_iter()
        self.valid_iter_initializer = valid_iter.initializer
        valid_features = valid_iter.get_next()
        if self.hp.label_type == 'arousal':
            valid_labels = valid_features['arousal']
        else:
            valid_labels = valid_features['valance']
        self.valid_model = model_func(self.hp, valid_features['spec'],
                                      valid_features['seq_len'],
                                      valid_labels)

        self.sess.run(tf.global_variables_initializer())

    def eval_valid(self):
        b_num = 0
        total_loss = 0
        prs = []
        gts = []

        sess = self.sess
        sess.run(self.valid_iter_initializer)
        MAX_LOOP = 9999
        for _ in range(MAX_LOOP):
            try:
                batched_logits, batched_gt, loss = sess.run(
                    (self.valid_model.logits, self.valid_model.labels,
                     self.valid_model.loss),
                    feed_dict={
                        self.valid_model.fc_kprob_ph: 1.0,
                        self.valid_model.is_training_ph: False
                    })
                b_num += 1
                total_loss += np.sum(loss)
                batched_pr = np.argmax(batched_logits, axis=-1)
                prs += list(batched_pr)
                gts += list(batched_gt)
            except tf.errors.OutOfRangeError:
                break
        valid_loss = total_loss / b_num
        # acc, precision, recall, f1 = eval_util.get_metric(prs, gts,
        #                                                   is_print=False)
        uar = eval_util.get_uar(prs, gts)
        return valid_loss, uar

    def train(self):

        # initialize ckpt related
        if not os.path.exists(self.hp.ckpt_dir):
            os.makedirs(self.hp.ckpt_dir)
        if not os.path.exists(self.hp.best_params_ckpt_dir):
            os.makedirs(self.hp.best_params_ckpt_dir)
        max_to_keep = 20
        if 'saver_max_to_keep' in self.hp:
            max_to_keep = self.hp.saver_max_to_keep
        saver1 = tf.train.Saver(max_to_keep=max_to_keep)
        saver2 = tf.train.Saver(max_to_keep=max_to_keep)
        best_uar = 0

        sess = self.sess
        sess.run(self.train_iter_initializer)

        # warm up run
        for i in range(self.hp.warm_up_steps + 1):
            sess.run(self.train_model.train_op,
                     feed_dict={
                         self.train_model.fc_kprob_ph: 1.0,
                         self.train_model.is_training_ph: True,
                         self.train_model.lr_ph: self.hp.warm_up_lr
                     })

        # sess.run
        for i in range(self.hp.warm_up_steps + 1,
                       self.hp.warm_up_steps + self.hp.train_steps + 1):
            if i % self.hp.train_print_interval == 0:
                _, loss, = sess.run(
                    (self.train_model.train_op, self.train_model.loss),
                    feed_dict={
                        self.train_model.fc_kprob_ph: 1.0,
                        self.train_model.is_training_ph: True,
                        self.train_model.lr_ph: self.hp.lr
                    })
                print('train:', i, loss)
            else:
                sess.run(self.train_model.train_op,
                         feed_dict={
                             self.train_model.fc_kprob_ph: 1.0,
                             self.train_model.is_training_ph: True,
                             self.train_model.lr_ph: self.hp.lr
                         })
            if i % self.hp.eval_interval == 0:
                valid_loss, uar = self.eval_valid()
                print('step {} , valid_loss {} , uar {}'.format(i, valid_loss,
                                                                uar))
                if uar > best_uar:
                    best_uar = uar
                    saver1.save(sess, save_path=os.path.join(
                        self.hp.best_params_ckpt_dir,
                        'model'), global_step=i)
            if i % self.hp.ckpt_interval == 0:
                saver2.save(sess, save_path=os.path.join(self.hp.ckpt_dir,
                                                         'model'),
                            global_step=i)

    def run(self):
        self.train()
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
    model_train = ModelTrain(yparams)
    model_train.run()


if __name__ == '__main__':
    main()
