from utils import parser_util, cfg_process
from scripts_crosslingual import prepare_data

if __name__ == '__main__':
    yparams = cfg_process.YParams('./gst_cfg/prepare_data.yml', 'default')
    parser = parser_util.MyArgumentParser()
    argc, flags_dict = parser.parse_to_dict()
    hpsPrecessor = cfg_process.HpsGSTPreprocessor(yparams, flags_dict)
    hpsPrecessor.preprocess()
    prepare_data.process(yparams)
