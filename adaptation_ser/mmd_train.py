import tensorflow as tf
from cross_lingual import d_set, eval_util, mmd_model
from utils import parser_util, cfg_process
import numpy as np
import os


class ModelTrain(object):

    def __init__(self, hp):
        self.hp = hp
        self.sess = tf.Session()

        if self.hp.model_type == 'gst2mmd':
            model_func = mmd_model.GST2MMDModel
        elif self.hp.model_type == 'mmd2':
            model_func = mmd_model.MMDmodel2
        else:
            model_func = mmd_model.MMDModel
        y_type = self.hp.label_type

        s_ds = d_set.DataSet(is_repeat=True, tfr_path=self.hp.train_tfr,
                             hp=self.hp)
        s_iter = s_ds.get_iter()
        self.s_iter_initializer = s_iter.initializer
        s_fs = s_iter.get_next()

        t_ds = d_set.DataSet(is_repeat=True, tfr_path=self.hp.eval_tfr,
                             hp=self.hp)
        t_iter = t_ds.get_iter()
        self.t_iter_initializer = t_iter.initializer
        t_fs = t_iter.get_next()

        self.train_model = model_func(self.hp, s_fs['spec'], s_fs['seq_len'],
                                      s_fs[y_type], t_fs['spec'],
                                      t_fs['seq_len'])

        valid_ds = d_set.DataSet(is_repeat=False, tfr_path=self.hp.valid_tfr,
                                 hp=self.hp)
        valid_iter = valid_ds.get_iter()
        self.valid_iter_initializer = valid_iter.initializer
        valid_fs = valid_iter.get_next()
        self.valid_model = model_func(self.hp, valid_fs['spec'],
                                      valid_fs['seq_len'], valid_fs[y_type],
                                      t_fs['spec'],
                                      t_fs['seq_len'])
        self.sess.run(tf.global_variables_initializer())

    def eval_valid(self):
        b_num = 0
        ce_loss = 0
        mmd_loss = 0
        total_loss = 0
        prs = []
        gts = []
        sess = self.sess
        sess.run(self.valid_iter_initializer)
        MAX_LOOP = 9999
        for _ in range(MAX_LOOP):
            try:
                b_logits, b_gt, b_ce, b_mmd, b_loss = sess.run(
                    (self.valid_model.logits, self.valid_model.s_y,
                     self.valid_model.loss_d['ce'],
                     self.valid_model.loss_d['mmd'],
                     self.valid_model.loss_d['total']),
                    feed_dict={
                        self.valid_model.fc_kprob_ph: 1.0,
                        self.valid_model.is_training_ph: False
                    })
                b_num += 1
                ce_loss += b_ce
                mmd_loss += b_mmd
                total_loss += b_loss
                b_pr = np.argmax(b_logits, axis=-1)
                prs += list(b_pr)
                gts += list(b_gt)
            except tf.errors.OutOfRangeError:
                break
        ce_loss /= b_num
        mmd_loss /= b_num
        total_loss /= b_num
        ua, wa = eval_util.get_ua_wa(prs, gts, is_print=False, result_f=None)
        print('ua:%.2f,wa:%.2f,ce:%.3f,mmd:%.3f,loss:%.3f' % (
            ua, wa, ce_loss, mmd_loss, total_loss))
        return total_loss, ua

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
        sess.run(self.s_iter_initializer)
        sess.run(self.t_iter_initializer)

        # warm up run
        for i in range(self.hp.warm_up_steps + 1):
            sess.run(self.train_model.train_op,
                     feed_dict={
                         self.train_model.fc_kprob_ph: 1.0,
                         self.train_model.is_training_ph: self.hp.k_prob,
                         self.train_model.lr_ph: self.hp.warm_up_lr
                     })

        # sess.run
        for i in range(self.hp.warm_up_steps + 1,
                       self.hp.warm_up_steps + self.hp.train_steps + 1):
            if i % self.hp.train_print_interval == 0:
                _, ce, mmd, t_loss = sess.run(
                    (self.train_model.train_op, self.train_model.loss_d['ce'],
                     self.train_model.loss_d['mmd'],
                     self.train_model.loss_d['total']),
                    feed_dict={
                        self.train_model.fc_kprob_ph: self.hp.k_prob,
                        self.train_model.is_training_ph: True,
                        self.train_model.lr_ph: self.hp.lr
                    }
                )
                print('train: step %d, ce %.3f, mmd %.3f, total %.3f' % (
                    i, ce, mmd, t_loss))
            else:
                sess.run(self.train_model.train_op,
                         feed_dict={
                             self.train_model.fc_kprob_ph: 1.0,
                             self.train_model.is_training_ph: True,
                             self.train_model.lr_ph: self.hp.lr
                         })
            if i % self.hp.eval_interval == 0:
                print('valid: step %d,' % i, end='')
                valid_loss, uar = self.eval_valid()
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
                        default='./mmd_cfg/recola2iemocap.yml',
                        help='config file about hparams')
    parser.add_argument('--config_name', type=str, default='default',
                        help='config name for hparams')


def main():
    parser = parser_util.MyArgumentParser()
    add_arguments(parser)
    argc, flags_dict = parser.parse_to_dict()
    yparams = cfg_process.YParams(argc.config_file, argc.config_name)
    yparams = cfg_process.HpsGSTPreprocessor(yparams, flags_dict).preprocess()
    yparams.json_save()
    model_train = ModelTrain(yparams)
    model_train.run()


if __name__ == '__main__':
    main()
