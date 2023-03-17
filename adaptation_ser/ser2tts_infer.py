import tensorflow as tf
from ser2tts import ser2tts_dset, ser2tts_model
from utils import parser_util, cfg_process


class ModelInfer(object):
    def __init__(self, hp):
        self.hp = hp
        self.sess = tf.Session()

        # model_func = ser2tts_model.MMDModel2

        ds = ser2tts_dset.TargetDataSet(tfr_dir=hp.target_tfr_dir,
                                        hp=hp,
                                        is_repeat=False)
        ds_iter = ds.get_iter()
        self.ds_init = ds_iter.initializer
        features = ds_iter.get_next()

        if hp.model_type == 'MMDBinary':
            self.model = ser2tts_model.MMDBinary(hp=hp, src_feature=features, tgt_feature=None,
                                                 label_type=hp.label_type, is_training=False, is_evaluation=False)
        else:
            self.model = ser2tts_model.MMDModel2(hp=hp, src_feature=features, tgt_feature=None,
                                                 is_training=False, is_evaluation=False)
        saver = tf.train.Saver()
        saver.restore(self.sess, self.hp.restore_ckpt)

    def predict(self):
        outputs_list = []
        uid_list = []
        MAX_LOOP = 9999
        self.sess.run(self.ds_init)
        for i in range(MAX_LOOP):
            try:
                batched_uid, batched_outputs = self.sess.run(
                    (self.model.src_feature['uid'], self.model.outputs)
                )
                print('step %d' % i)
                uid_list += list(batched_uid)
                outputs_list += list(batched_outputs)
            except tf.errors.OutOfRangeError:
                break
        assert len(uid_list) == len(outputs_list)
        with open(self.hp.result_file, 'w') as out_f:
            for uid, outputs in zip(uid_list, outputs_list):
                uid_decode = uid.decode('utf-8')
                print(uid_decode)
                print(uid_decode, '|', list(outputs), file=out_f)

    def run(self):
        self.predict()
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
    model_infer = ModelInfer(yparams)
    model_infer.run()


if __name__ == '__main__':
    main()
