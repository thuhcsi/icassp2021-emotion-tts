import os
import shutil
import numpy as np

emo2idx = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3, 'exc': 2}

sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']


def process():
    # in_base_dir = '/Users/ddy/Projects/data/IEMOCAP_full_release'
    in_base_dir = '/home/ddy17/ser_data/iemocap/IEMOCAP_full_release'
    eval_sub_fold = 'dialog/EmoEvaluation'
    out_base_dir = '/home/ddy17/ser_data/ser2tts/iemo'
    os.makedirs(out_base_dir, exist_ok=True)
    all_meta_list = []
    for sess in sessions:
        print(sess)
        session_emo_eval_dir = os.path.join(in_base_dir, sess, eval_sub_fold)
        meta_path = os.path.join(out_base_dir, sess + '.txt')
        meta_list = process_session(session_emo_eval_dir, meta_path)
        all_meta_list += meta_list
    out_path = os.path.join(out_base_dir, 'meta.txt')
    with open(out_path, 'w') as out_f:
        for meta_line in all_meta_list:
            print(meta_line, file=out_f)


def process_session(session_emo_eval_dir, meta_path):
    file_names = os.listdir(session_emo_eval_dir)
    emo_file_names = [f_name for f_name in file_names if 'Ses' in f_name and '.txt' in f_name]
    meta_list = []
    for emo_file_name in emo_file_names:
        emo_path = os.path.join(session_emo_eval_dir, emo_file_name)
        process_file(emo_path, meta_list)
    with open(meta_path, 'w') as out_f:
        for meta_line in meta_list:
            print(meta_line, file=out_f)
    return meta_list


def process_file(eval_file_path, meta_list):
    with open(eval_file_path, 'r') as inf:
        for line in inf:
            if '[' in line and ']' in line and 'Ses' in line:
                eles = line.strip().split()
                if eles[4] in emo2idx.keys():
                    idx_str = str(emo2idx[eles[4]])
                    meta_line = '|'.join((eles[3], eles[4], idx_str))
                    meta_list.append(meta_line)


def get_path_from_wav_id(wav_id):
    base_in_dir = '/home/ddy17/ser_data/iemocap/IEMOCAP_full_release/wavs'

    wav_path = os.path.join(base_in_dir, wav_id + '.wav')
    return wav_path


def copy_wavs():
    base_in_dir = '/home/ddy17/ser_data/iemocap/IEMOCAP_full_release'
    out_fold = 'wavs'
    out_dir = os.path.join(base_in_dir, out_fold)
    for sess in sessions:
        sess_wav_dir = os.path.join(base_in_dir, sess, 'sentences/wav')
        fold_names = [f for f in os.listdir(sess_wav_dir) if 'Ses' in f]
        for fold_name in fold_names:
            fold_path = os.path.join(sess_wav_dir, fold_name)
            file_names = [f for f in os.listdir(fold_path) if 'Ses' in f]
            for file_name in file_names:
                file_path = os.path.join(fold_path, file_name)
                out_path = os.path.join(out_dir, file_name)
                shutil.copy2(file_path, out_path)


def calc_emo():
    emo_num = 2
    emo_count = np.zeros(emo_num)
    impro_emo_count = np.zeros(emo_num)
    meta_file = '/home/ddy17/ser_data/cx_data/iemocap/meta_valance'
    with open(meta_file, 'r') as meta_f:
        for line in meta_f:
            eles = line.strip().split('|')
            if len(eles) == 3:
                emo_idx = int(eles[2])
                emo_count[emo_idx] += 1
                if 'impro' in line:
                    impro_emo_count[emo_idx] += 1
    return list(emo_count), list((1. / emo_count) / np.mean(1. / emo_count)), list(impro_emo_count), list(
        (1. / impro_emo_count) / np.mean(1. / impro_emo_count))


def select_meta_wavs():
    meta_file = '/home/ddy17/ser_data/ser2tts/iemo/meta.txt'
    wavs_dir = '/home/ddy17/ser_data/ser2tts/iemo/wavs'
    with open(meta_file, 'r') as meta_f:
        for line in meta_f:
            eles = line.strip().split('|')
            if len(eles) == 3:
                wav_id = eles[0]
                wav_path = get_path_from_wav_id(wav_id)
                out_path = os.path.join(wavs_dir, wav_id + '.wav')
                shutil.copy2(wav_path, out_path)


def test():
    eval_file_path = '/Users/ddy/Projects/data/IEMOCAP_full_release/Session1/dialog/EmoEvaluation/Ses01F_impro02.txt'


arousal_dict = {'low': 0, 'hig': 1}
valance_dict = {'neg': 0, 'pos': 1}


def process_cx_meta(cx_meta_path, out_arousal_path, out_valance_path):
    cx_f = open(cx_meta_path, 'r')
    out_arousal_f = open(out_arousal_path, 'w')
    out_valance_f = open(out_valance_path, 'w')
    for cx_line in cx_f:
        cx_eles = cx_line.strip().split('|')
        if len(cx_eles) == 5:
            uid = cx_eles[0].split('/')[-1].strip('.wav')
            arousal_origin = float(cx_eles[3])
            if arousal_origin <= 2.51:
                ar_note = 'low'
                ar_v = str(0)
            else:
                ar_note = 'hig'
                ar_v = str(1)
            print('|'.join((uid, ar_note, ar_v)), file=out_arousal_f)
            valance_origin = float(cx_eles[4])
            if valance_origin <= 2.51:
                va_note = 'neg'
                va_v = str(0)
            else:
                va_note = 'pos'
                va_v = str(1)
            print('|'.join((uid, va_note, va_v)), file=out_valance_f)
    cx_f.close()
    out_arousal_f.close()
    out_valance_f.close()


def main_process_cx_meta():
    cx_meta_path = '/home/ddy17/ser_data/cx_data/iemocap/wav_meta'
    out_arousal_path = '/home/ddy17/ser_data/cx_data/iemocap/meta_arousal'
    out_valance_path = '/home/ddy17/ser_data/cx_data/iemocap/meta_valance'
    process_cx_meta(cx_meta_path, out_arousal_path, out_valance_path)


if __name__ == '__main__':
    print(calc_emo())
    # main_process_cx_meta()
    # process()
    # copy_wavs()
    # select_meta_wavs()
    # copy_wavs()
    # process()
    # select_meta_wavs()
    # print(calc_emo())
