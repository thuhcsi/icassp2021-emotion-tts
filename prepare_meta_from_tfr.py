import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from text import sequence_to_text
from tfr_dset import load_for_prepare_meta_from_tfr


max_steps = 100000   # actual 95650
tfr_dir = 'bc2013/training/tfrs_with_emo_feature'
meta_path = 'bc2013/full_meta.txt'
mel_path = 'bc2013/mels'
spec_path = 'bc2013/specs'


def main():
    tf_dset = load_for_prepare_meta_from_tfr(tfr_dir)
    feats = tf_dset.make_one_shot_iterator().get_next()

    i, lines, mels, specs = 1, [], [], []
    lines.append('# emo|aro|val|text_len|spec_len|text|uid|mel|spec\n')

    pbar = tqdm(max_steps)
    sess = tf.Session()
    try:
        while True:
            fetched_feats = sess.run(feats)
            # uid = fetched_feats['uid'].tobytes().decode('utf-8')
            uid = fetched_feats['uid'].decode('utf-8')
            text = sequence_to_text(fetched_feats['inputs'])
            text_lens = fetched_feats['input_lengths']
            mel = fetched_feats['mel_targets']
            spec = fetched_feats['linear_targets']
            spec_lens = fetched_feats['spec_lengths']
            emo = '[{:.5f}, {:.5f}, {:.5f}, {:.5f}]'.format(*fetched_feats['soft_emo_labels'])
            aro = '[{:.5f}, {:.5f}]'.format(*fetched_feats['soft_arousal_labels'])
            val = '[{:.5f}, {:.5f}]'.format(*fetched_feats['soft_valance_labels'])
            mel_name = os.path.join(mel_path, f'bc13-mel-{i:06d}.npy')
            spec_name = os.path.join(spec_path, f'bc13-spec-{i:06d}.npy')
            line = f'{emo}|{aro}|{val}|{text_lens}|{spec_lens}|{text}|{uid}|{mel_name}|{spec_name}\n'
            lines.append(line)
            mels.append([mel, mel_name])
            specs.append([spec, spec_name])
            pbar.update(1)
            i += 1
    except tf.errors.OutOfRangeError:
        print('sess.run finished!')
    finally:
        pbar.close()
        sess.close()

    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(spec_path, exist_ok=True)
    for mel, mel_name in tqdm(mels):
        np.save(mel_name, mel)
    for spec, spec_name in tqdm(specs):
        np.save(spec_name, spec)
    with open(meta_path, 'w') as fw:
        fw.writelines(lines)
    print(f'total {i} items finished!')


if __name__ == '__main__':
    main()
