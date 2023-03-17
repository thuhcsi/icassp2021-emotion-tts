# this command is used for extract mel features
import os

import librosa
import numpy as np
import argparse

n_fft = 1024
win_time = 0.04
hop_time = 0.01


def get_log_mel_spectrogram(y, sr, n_fft, win_length, hop_length, power=2, window='hamming',
                            n_mels=128):
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                      window=window)) ** power
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=n_mels)
    mel_s = np.dot(mel_basis, spectrogram)
    log_mel_s = librosa.core.power_to_db(mel_s)
    return log_mel_s.transpose()


def process(base_dir, max_frames_len=1400, classify_point=0.):
    wav_meta_path = os.path.join(base_dir, 'wav_meta')
    npy_meta_path = os.path.join(base_dir, 'npy_meta')
    wav_dir = os.path.join(base_dir, 'wavs')
    npy_dir = os.path.join(base_dir, 'npys')
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    npy_metas = []
    arousal_count = 0
    valance_count = 0
    count = 0
    with open(wav_meta_path, 'r') as wav_meta_f:
        wav_metas = wav_meta_f.readlines()
        for wav_i_str in wav_metas:
            wav_i = [item.strip() for item in wav_i_str.strip().split('|')]
            wav_name = wav_i[0].split('/')[-1]
            arousal = 0 if float(wav_i[-2]) <= classify_point else 1
            valance = 0 if float(wav_i[-1]) <= classify_point else 1
            arousal_count += arousal
            valance_count += valance
            count += 1
            # npy_metas.append((wav_name, arousal, valance))
            wav_path = os.path.join(wav_dir, wav_name)
            npy_name = os.path.splitext(wav_name)[0] + '.npy'
            npy_metas.append((npy_name, str(arousal), str(valance)))
            npy_path = os.path.join(npy_dir, npy_name)
            y, sr = librosa.load(wav_path, sr=16000)
            win_length = int(win_time * sr)
            hop_length = int(hop_time * sr)

            log_s = get_log_mel_spectrogram(y=y, sr=sr, n_fft=n_fft, win_length=win_length,
                                            hop_length=hop_length, power=2,
                                            window='hamming',
                                            n_mels=128)
            frames_len = log_s.shape[0]
            if frames_len > max_frames_len:
                # print(frames_len)
                # print(max_frames_len)
                # print(frames_len // 2 - max_frames_len // 2)
                log_s = log_s[frames_len // 2 - max_frames_len // 2: frames_len // 2 + max_frames_len // 2, :]
            print(wav_name, log_s.shape)
            np.save(npy_path, log_s)
    print('arousal positive', arousal_count)
    print('valance positive', valance_count)
    print('example count', count)
    with open(npy_meta_path, 'w') as npy_meta_f:
        for npy_meta in npy_metas:
            print('|'.join(npy_meta), file=npy_meta_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        help='base dir of wavs fold and npys fold')
    parser.add_argument('--classify_point',
                        type=float,
                        default=0.)
    parser.add_argument('--max_frames_len',
                        type=int,
                        default=1400)
    args = parser.parse_args()
    process(args.base_dir, args.max_frames_len, args.classify_point)
