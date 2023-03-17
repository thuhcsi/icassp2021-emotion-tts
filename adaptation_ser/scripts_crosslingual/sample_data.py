import argparse
import random
import os
import shutil


def write_list(out_path, content_list):
    with open(out_path, 'w') as out_f:
        for content in content_list:
            print('| '.join(content), file=out_f)


def process_sub_dataset(in_dataset_dir, out_dataset_dir,  sample_rate):
    if not os.path.exists(os.path.join(out_dataset_dir, 'wavs')):
        os.makedirs(os.path.join(out_dataset_dir, 'wavs'), exist_ok=True)
    meta_path = os.path.join(in_dataset_dir, 'wav_meta')
    out_list = []
    with open(meta_path, 'r') as in_f:
        metas = in_f.readlines()
        metas = [meta.split('|') for meta in metas]
        metas = [('/'.join(meta[0].split('/')[-2:]), meta[-2], meta[-1].strip()) for meta in metas]
        random.shuffle(metas)
        for i, meta in enumerate(metas):
            if i % sample_rate == 0:
                out_list.append(meta)
    for out_item in out_list:
        src_wav = os.path.join(in_dataset_dir, out_item[0])
        tgt_wav = os.path.join(out_dataset_dir, out_item[0])
        shutil.copy2(src_wav, tgt_wav)
    write_list(os.path.join(out_dataset_dir, 'wav_meta'), out_list)


def main():
    data_folds = ['recola']
    # data_folds = ['iemocap', 'recola']
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default='/Users/bytedance/Projects/data/cx_data',
                        help='data directory of RECOLA data and IEMOCAP data')
    parser.add_argument('--out_dir',
                        default='/Users/bytedance/Projects/data/cx_data_simple')
    parser.add_argument('--sample_rate',
                        default=200,
                        help='every N sample 1')

    args = parser.parse_args()
    for data_fold in data_folds:
        in_dataset_dir = os.path.join(args.data_dir, data_fold)
        out_dataset_dir = os.path.join(args.out_dir, data_fold)
        process_sub_dataset(in_dataset_dir, out_dataset_dir, args.sample_rate)


if __name__ == '__main__':
    main()
