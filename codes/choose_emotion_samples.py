import os
import json
import shutil
import numpy as np

meta_file = 'bc2013/full_meta.txt'
wavs_path = 'bc2013/wavs'
mels_path = 'bc2013/mels'
specs_path = 'bc2013/specs'


def save_topk_files(saved_wavs_path, saved_mels_path, meta_path, lines):
    # copy wavs and mels
    for i, line in enumerate(lines):
        wav_name = line[6] + '.wav'
        mel_name = os.path.basename(line[7])
        source_wav_name = os.path.join(wavs_path, wav_name)
        source_mel_name = os.path.join(mels_path, mel_name)
        saved_wav_name = os.path.join(saved_wavs_path, wav_name)
        saved_mel_name = os.path.join(saved_mels_path, f'{i:03d}-' + mel_name)  # add a sorted prefix
        os.makedirs(os.path.dirname(saved_wav_name), exist_ok=True)
        os.makedirs(os.path.dirname(saved_mel_name), exist_ok=True)
        shutil.copy(source_wav_name, saved_wav_name)
        shutil.copy(source_mel_name, saved_mel_name)

    # save meta lines
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    lines = ['{}|{}|{}|{}|{}|{}|{}|{}|{}\n'.format(*line) for line in lines]
    with open(meta_path, 'w') as fw:
        fw.writelines(lines)


def choose_emo_samples(meta_lines, base_dir='tmp', top_k=100, min_chars=50, min_frames=200):
    mels_path = os.path.join(base_dir, 'emo_mel_npys/emo{}')
    wavs_path = os.path.join(base_dir, 'emo_wavs/emo{}')
    meta_path = os.path.join(base_dir, 'emo_metas/emo{}.txt')
    for i in range(4):
        topk_lines = []
        # line[0] is the 4 categories of soft label: [neutral, angry, happy, sad]
        sorted_lines = sorted(meta_lines, key=lambda x: x[0][i], reverse=True)
        for line in sorted_lines:
            if line[3] >= min_chars and line[4] >= min_frames:
                if i == 0:
                    topk_lines.append(line)
                # elif i == 1 and line[1][1] > 0.5 and line[2][0] > 0.5:    # angry
                elif i == 1 and line[1][1] > 0.5 and line[2][0] > 0.45:    # angry
                    topk_lines.append(line)
                elif i == 2 and line[1][1] > 0.5 and line[2][1] > 0.45:  # happy
                    topk_lines.append(line)
                # elif i == 3 and line[1][0] > 0.5 and line[2][0] > 0.5:  # sad
                elif i == 3 and line[1][0] > 0.40 and line[2][0] > 0.45:  # sad
                    topk_lines.append(line)
                if len(topk_lines) == top_k:
                    break
        save_topk_files(wavs_path.format(i), mels_path.format(i), meta_path.format(i), topk_lines)


def choose_aro_samples(meta_lines, base_dir='tmp', top_k=100, min_chars=50, min_frames=200):
    mels_path = os.path.join(base_dir, 'emo2d_mel_npys/arousal{}')
    wavs_path = os.path.join(base_dir, 'emo2d_wavs/arousal{}')
    meta_path = os.path.join(base_dir, 'emo2d_metas/arousal{}.txt')
    for i in range(2):
        topk_lines = []
        # line[1] is the arousal soft label
        sorted_lines = sorted(meta_lines, key=lambda x: x[1][i], reverse=True)
        for line in sorted_lines:
            if line[3] >= min_chars and line[4] >= min_frames:
                if i == 0 and np.argmax(line[0]) in [0, 3]:
                    topk_lines.append(line)
                elif i == 1 and np.argmax(line[0]) in [1, 2]:
                    topk_lines.append(line)
                if len(topk_lines) == top_k:
                    break
        save_topk_files(wavs_path.format(i), mels_path.format(i), meta_path.format(i), topk_lines)


def choose_val_samples(meta_lines, base_dir='tmp', top_k=100, min_chars=50, min_frames=200):
    mels_path = os.path.join(base_dir, 'emo2d_mel_npys/valence{}')
    wavs_path = os.path.join(base_dir, 'emo2d_wavs/valence{}')
    meta_path = os.path.join(base_dir, 'emo2d_metas/valence{}.txt')
    for i in range(2):
        topk_lines = []
        # line[2] is the valence soft label
        sorted_lines = sorted(meta_lines, key=lambda x: x[2][i], reverse=True)
        for line in sorted_lines:
            if line[3] >= min_chars and line[4] >= min_frames:
                if i == 0 and np.argmax(line[0]) in [1, 3]:
                    topk_lines.append(line)
                elif i == 1 and np.argmax(line[0]) in [0, 2]:
                    topk_lines.append(line)
                if len(topk_lines) == top_k:
                    break
        save_topk_files(wavs_path.format(i), mels_path.format(i), meta_path.format(i), topk_lines)


def main():
    with open(meta_file, 'r') as fr:
        meta_lines = fr.readlines()

    def parse_emostrs(line):
        line[0] = json.loads(line[0])  # emotion 4 categories label
        line[1] = json.loads(line[1])  # arousal label
        line[2] = json.loads(line[2])  # valence label
        line[3] = int(line[3])  # text length
        line[4] = int(line[4])  # mel frame number
        return line
    meta_lines = [line.strip().split('|') for line in meta_lines if line.strip()[0] != '#']
    meta_lines = list(map(parse_emostrs, meta_lines))

    """
    choose_emo_samples(meta_lines, 'emogst_emo_data')
    choose_aro_samples(meta_lines, 'sygst_emo_data')
    choose_val_samples(meta_lines, 'sygst_emo_data')
    """
    choose_emo_samples(meta_lines, top_k=200)
    choose_aro_samples(meta_lines, top_k=200)
    choose_val_samples(meta_lines, top_k=200)


if __name__ == '__main__':
    main()
