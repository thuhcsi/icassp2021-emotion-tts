import tensorflow as tf
from ser2tts import ser2tts_dset, ser2tts_model
from utils import parser_util, cfg_process
import numpy as np
import os
from sklearn.metrics import recall_score


class ModelTrain(object):

    def __init__(self, hp):
        self.hp = hp
        self.sess = tf.Session()

        # if hp.model_type == 'MMDBinary':
        #     model_func = ser2tts_model.MMDBinary

        s_ds = ser2tts_dset.SourceDataSet(tfr_dir=hp.train_tfr_dir,
                                          hp=hp, is_repeat=True)
        train_iter = s_ds.get_iter()
        self.train_iter_init = train_iter.initializer
        train_feature = train_iter.get_next()
        self.train_feature = train_feature

        t_ds = ser2tts_dset.TargetDataSet(tfr_dir=hp.target_tfr_dir,
                                          hp=hp, is_repeat=True)
        t_iter = t_ds.get_iter()
        self.t_iter_init = t_iter.initializer
        tgt_feature = t_iter.get_next()

        if hp.model_type == 'MMDBinary':
            self.train_model = ser2tts_model.MMDBinary(hp=hp, src_feature=train_feature, tgt_feature=tgt_feature,
                                                       label_type=hp.label_type, is_training=True, is_evaluation=False)
        else:
            self.train_model = ser2tts_model.MMDModel2(hp=hp, src_feature=train_feature, tgt_feature=tgt_feature,
                                                       is_training=True, is_evaluation=False)

        valid_ds = ser2tts_dset.SourceDataSet(tfr_dir=hp.valid_tfr_dir, hp=hp, is_repeat=False)
        valid_iter = valid_ds.get_iter()
        self.valid_iter_init = valid_iter.initializer
        valid_feature = valid_iter.get_next()
        self.valid_feature = valid_feature
        if hp.model_type == 'MMDBinary':
            self.valid_model = ser2tts_model.MMDBinary(hp=hp, src_feature=valid_feature, tgt_feature=tgt_feature,
                                                       label_type=hp.label_type, is_training=False, is_evaluation=True)
        else:
            self.valid_model = ser2tts_model.MMDModel2(hp=hp, src_feature=valid_feature, tgt_feature=tgt_feature,
                                                       is_training=False, is_evaluation=True)

        self.sess.run(tf.global_variables_initializer())

    def eval_valid(self):
        b_num = 0
        ce_loss = 0
        mmd_loss = 0
        total_loss = 0
        prs = []
        gts = []
        sess = self.sess
        sess.run(self.valid_iter_init)
        MAX_LOOP = 9999
        for _ in range(MAX_LOOP):
            try:
                b_outputs, b_gt, b_ce, b_mmd, b_loss = sess.run(
                    (self.valid_model.outputs,
                     self.valid_model.src_feature['emo_idx'],
                     self.valid_model.loss_d['ce'],
                     self.valid_model.loss_d['mmd'],
                     self.valid_model.loss_d['total']))
                b_num += 1
                ce_loss += b_ce
                mmd_loss += b_mmd
                total_loss += b_loss
                b_pr = np.argmax(b_outputs, axis=-1).astype(np.int32)
                prs += list(b_pr)
                gts += list(b_gt)
            except tf.errors.OutOfRangeError:
                break
        ce_loss /= b_num
        mmd_loss /= b_num
        total_loss /= b_num
        gt_np = np.array(gts)
        pr_np = np.array(prs)
        ua = recall_score(y_true=gt_np, y_pred=pr_np, average='macro')
        wa = float(np.sum((gt_np == pr_np).astype(np.float)) / len(gt_np))
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
        sess.run(self.train_iter_init)
        sess.run(self.t_iter_init)
        # warm up run
        for i in range(self.hp.warm_up_steps + 1):
            sess.run(self.train_model.train_op,
                     feed_dict={
                         self.train_model.lr_ph: self.hp.warm_up_lr
                     })
        for i in range(self.hp.warm_up_steps + 1,
                       self.hp.warm_up_steps + self.hp.train_steps + 1):
            if i % self.hp.train_print_interval == 0:
                if i % self.hp.train_print_interval == 0:
                    _, ce, mmd, t_loss = sess.run(
                        (self.train_model.train_op, self.train_model.loss_d['ce'],
                         self.train_model.loss_d['mmd'],
                         self.train_model.loss_d['total']),
                        feed_dict={
                            self.train_model.lr_ph: self.hp.lr
                        }
                    )
                    print('train: step %d, ce %.3f, mmd %.3f, total %.3f' % (
                        i, ce, mmd, t_loss))
                else:
                    sess.run(self.train_model.train_op,
                             feed_dict={
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
                        default='./ser2tts_cfg/iemo2bc.yml',
                        help='config file about hparams')
    parser.add_argument('--config_name', type=str, default='default',
                        help='config name for hparams')


def main():
    parser = parser_util.MyArgumentParser()
    add_arguments(parser)
    argc, flags_dict = parser.parse_to_dict()
    yparams = cfg_process.YParams(argc.config_file, argc.config_name)
    yparams = cfg_process.HpsSER2TTSPreprocessor(yparams, flags_dict).preprocess()
    yparams.json_save()
    model_train = ModelTrain(yparams)
    model_train.run()
    # model_train.test_set()


if __name__ == '__main__':
    main()
