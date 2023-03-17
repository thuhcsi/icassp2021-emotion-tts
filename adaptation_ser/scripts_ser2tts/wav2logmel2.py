# this command is used for extract mel features
import os

import librosa
import numpy as np
import argparse

n_fft = 1024
win_time = 0.04
hop_time = 0.01
sr = 16000


def get_log_mel_spectrogram(y, sr, n_fft, win_length, hop_length, power=2, window='hamming',
                            n_mels=128):
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                      window=window)) ** power
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=n_mels)
    mel_s = np.dot(mel_basis, spectrogram)
    log_mel_s = librosa.core.power_to_db(mel_s)
    return log_mel_s.transpose()


def load_wav(wav_path, sr, norm_wav=True):
    y, sr = librosa.load(wav_path, sr=sr)
    if norm_wav:
        rescaling_max = 0.999
        y = y / np.abs(y).max() * rescaling_max
    return y, sr


def create_npy_dirs(wav_dir, npy_dir):
    os.makedirs(npy_dir, exist_ok=True)
    file_names = os.listdir(wav_dir)
    for file_name in file_names:
        wav_path = os.path.join(wav_dir, file_name)
        if os.path.isdir(wav_path):
            npy_path = os.path.join(npy_dir, file_name)
            os.makedirs(npy_path, exist_ok=True)


def wav2mel(wav_dir, npy_dir, meta_fpath, max_frames_len=1400):
    create_npy_dirs(wav_dir, npy_dir)
    with open(meta_fpath, 'r') as meta_f:
        for line in meta_f:
            eles = line.strip().split('|')
            if len(eles) == 3:
                wav_id = eles[0]
                wav_path = os.path.join(wav_dir, wav_id + '.wav')
                npy_path = os.path.join(npy_dir, wav_id + '.npy')
                y, _ = load_wav(wav_path, sr=sr)
                win_length = int(win_time * sr)
                hop_length = int(hop_time * sr)
                log_s = get_log_mel_spectrogram(y=y, sr=sr, n_fft=n_fft, win_length=win_length,
                                                hop_length=hop_length, power=2,
                                                window='hamming', n_mels=128)
                frames_len = log_s.shape[0]
                if frames_len > max_frames_len:
                    log_s = log_s[frames_len // 2 - max_frames_len // 2: frames_len // 2 + max_frames_len // 2, :]
                if frames_len < 20:
                    continue
                print(wav_path, log_s.shape)
                np.save(npy_path, log_s)


def arg_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        required=True,
                        help='base dir of wavs fold and npys fold')
    parser.add_argument('--meta_file',
                        default='meta.txt',
                        help='meta file name')
    parser.add_argument('--max_frames_len',
                        type=int,
                        default=1400)
    args = parser.parse_args()
    wav_dir = os.path.join(args.base_dir, 'wavs')
    npy_dir = os.path.join(args.base_dir, 'npys')
    meta_fpath = os.path.join(args.base_dir, args.meta_file)
    wav2mel(wav_dir, npy_dir, meta_fpath, max_frames_len=args.max_frames_len)


if __name__ == '__main__':
    arg_main()
