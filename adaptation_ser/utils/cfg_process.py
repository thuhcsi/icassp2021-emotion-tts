from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams
import os
import time
from utils import parser_util

__all__ = ["YParams", "HpsGSTPreprocessor"]


class YParams(HParams):
    def __init__(self, yaml_f, config_name):
        super().__init__()
        with open(yaml_f) as fp:
            for k, v in YAML().load(fp)[config_name].items():
                self.add_hparam(k, v)

    def save(self, filename=None):
        if filename is None:
            filename = self.get("id_str") + "_hparams.yml"
        if not os.path.exists(self.get("cfg_out_dir")):
            os.makedirs(self.get("cfg_out_dir"))
        file_path = os.path.join(self.get("cfg_out_dir"), filename)
        with open(file_path, "w") as out_f:
            YAML().dump(self.values(), out_f)

    def json_save(self, filename=None):
        if filename is None:
            filename = self.get("id_str") + "_hparams.json"
        if not os.path.exists(self.get("cfg_out_dir")):
            os.makedirs(self.get("cfg_out_dir"))
        file_path = os.path.join(self.get("cfg_out_dir"), filename)
        with open(file_path, "w") as out_f:
            out_f.write(self.to_json())


class HpsBasePreprocessor(object):
    def __init__(self, hparams, flags):
        self.hparams = hparams
        if flags is None:
            return
        for k, v in flags.items():
            if k in hparams:
                try:
                    hparams.set_hparam(k, v)
                except ValueError:
                    hparams.set_hparam(k, str(v))
            else:
                hparams.add_hparam(k, v)

    def _update_id_related(self):
        if 'id' not in self.hparams or self.hparams.id == '':
            self.hparams.id = time.strftime("%m%d%H%M", time.localtime())
        if 'id_prefix' not in self.hparams:
            id_str = self.hparams.id
        else:
            id_str = self.hparams.id_prefix + self.hparams.id
        if 'id_str' in self.hparams:
            self.hparams.id_str = id_str
        else:
            self.hparams.add_hparam('id_str', id_str)

    def _cuda_visiable_devices(self):
        if 'gpu' in self.hparams and self.hparams.gpu != '':
            if 'CUDA_VISIBLE_DEVICES' not in self.hparams:
                self.hparams.add_hparam('CUDA_VISIBLE_DEVICES',
                                        self.hparams.gpu)
            else:
                self.hparams.CUDA_VISIBLE_DEVICES = self.hparams.gpu
        if 'CUDA_VISIBLE_DEVICES' in self.hparams:
            os.environ[
                'CUDA_VISIBLE_DEVICES'] = self.hparams.CUDA_VISIBLE_DEVICES

    def preprocess(self):
        self._update_id_related()
        # self._check_dir()
        self._cuda_visiable_devices()
        return self.hparams


class HpsGSTPreprocessor(HpsBasePreprocessor):

    def _update_ckpt_related(self):
        prefix = '_'.join((self.hparams.model_type, self.hparams.label_type,
                           self.hparams.id_str))
        ckpt_fold = '_'.join((prefix, 'ckpt'))
        best_params_fold = '_'.join((prefix, 'best_params_ckpt'))
        ckpt_dir = os.path.join(self.hparams.ckpt_base_dir, ckpt_fold)
        best_params_ckpt_dir = os.path.join(self.hparams.ckpt_base_dir,
                                            best_params_fold)
        if ckpt_dir not in self.hparams:
            self.hparams.add_hparam('ckpt_dir', ckpt_dir)
        else:
            self.hparams.ckpt_dir = ckpt_dir

        if best_params_ckpt_dir not in self.hparams:
            self.hparams.add_hparam('best_params_ckpt_dir',
                                    best_params_ckpt_dir)
        else:
            self.hparams.best_params_ckpt_dir = best_params_ckpt_dir

    def preprocess(self):
        self._update_id_related()
        self._update_ckpt_related()
        self._cuda_visiable_devices()
        return self.hparams


class HpsSER2TTSPreprocessor(HpsBasePreprocessor):

    def _update_ckpt_related(self):
        prefix = '_'.join((self.hparams.model_type, self.hparams.id_str))
        ckpt_fold = '_'.join((prefix, 'ckpt'))
        best_params_fold = '_'.join((prefix, 'best_params_ckpt'))
        ckpt_dir = os.path.join(self.hparams.ckpt_base_dir, ckpt_fold)
        best_params_ckpt_dir = os.path.join(self.hparams.ckpt_base_dir,
                                            best_params_fold)
        if ckpt_dir not in self.hparams:
            self.hparams.add_hparam('ckpt_dir', ckpt_dir)
        else:
            self.hparams.ckpt_dir = ckpt_dir

        if best_params_ckpt_dir not in self.hparams:
            self.hparams.add_hparam('best_params_ckpt_dir',
                                    best_params_ckpt_dir)
        else:
            self.hparams.best_params_ckpt_dir = best_params_ckpt_dir

    def preprocess(self):
        self._update_id_related()
        self._update_ckpt_related()
        self._cuda_visiable_devices()
        return self.hparams


if __name__ == '__main__':
    yparams = YParams('./gst_cfg/test.yml', 'default')
    parser = parser_util.MyArgumentParser()
    argc, flags_dict = parser.parse_to_dict()
    hpsPrecessor = HpsGSTPreprocessor(yparams, flags_dict)
    yparams.save()
