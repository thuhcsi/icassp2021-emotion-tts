import librosa
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile


_mel_basis = None


def load_wav(hp, path):
    wav, sr = librosa.core.load(path, sr=hp.sample_rate)
    wav = wav / np.abs(wav).max() * 0.999
    return wav, sr


def save_wav(hp, wav, path):
    wav /= max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hp.sample_rate, (wav * 32766).astype(np.int16))


def preemphasis(hp, x):
    return signal.lfilter([1, -hp.preemphasis], [1], x)


def inv_preemphasis(hp, x):
    return signal.lfilter([1], [1, -hp.preemphasis], x)


def spectrogram(hp, y):
    if hp.preemphasis is None:
        D = _stft(hp, y)
    else:
        D = _stft(hp, preemphasis(hp, y))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    return _normalize(hp, S)


def inv_spectrogram(hp, spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(hp, spectrogram) + hp.ref_level_db)  # Convert back to linear
    return inv_preemphasis(hp, _griffin_lim(hp, S ** hp.griffin_lim_power))  # Reconstruct phase


def inv_spectrogram_tensorflow(hp, spectrogram):
    '''Builds computational graph to convert spectrogram to waveform using TensorFlow.

    Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
    inv_preemphasis on the output after running the graph.
    '''
    with tf.name_scope('griffin_lim'):
        S = _db_to_amp_tensorflow(_denormalize_tensorflow(hp, spectrogram) + hp.ref_level_db)
        return _griffin_lim_tensorflow(hp, tf.pow(S, hp.griffin_lim_power))


def melspectrogram(hp, y):
    if hp.preemphasis is None:
        D = _stft(hp, y)
    else:
        D = _stft(hp, preemphasis(hp, y))
    S = _amp_to_db(_linear_to_mel(hp, np.abs(D))) - hp.ref_level_db
    return _normalize(hp, S)


def mfcc(hp, y):
    pass


def find_endpoint(hp, wav, threshold_db=-40, min_silence_sec=0.8):
    window_length = int(hp.sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x + window_length]) < threshold:
            return x + hop_length
    return len(wav)


def _griffin_lim(hp, S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(hp, S_complex * angles)
    for i in range(hp.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(hp, y)))
        y = _istft(hp, S_complex * angles)
    return y   # reconstructed wav


def _griffin_lim_tensorflow(hp, S):
    '''TensorFlow implementation of Griffin-Lim
    Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
    '''
    with tf.variable_scope('griffinlim'):
        # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = _istft_tensorflow(hp, S_complex)
        for i in range(hp.griffin_lim_iters):
            est = _stft_tensorflow(hp, y)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = _istft_tensorflow(hp, S_complex * angles)
        return tf.squeeze(y, 0)


def _stft(hp, y):
    n_fft, hop_length, win_length = _stft_parameters(hp)
    # shape (1 + n_fft/2, n_frames)
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(hp, y):
    _, hop_length, win_length = _stft_parameters(hp)
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(hp, signals):
    n_fft, hop_length, win_length = _stft_parameters(hp)
    return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft_tensorflow(hp, stfts):
    n_fft, hop_length, win_length = _stft_parameters(hp)
    return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters(hp):
    n_fft = hp.n_fft
    hop_length = int(hp.hop_ms / 1000 * hp.sample_rate)
    win_length = int(hp.win_ms / 1000 * hp.sample_rate)
    return n_fft, hop_length, win_length


def _linear_to_mel(hp, spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hp)
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis(hp):
    n_fft = hp.n_fft
    return librosa.filters.mel(hp.sample_rate, n_fft, n_mels=hp.num_mels)


def _amp_to_db(x):
    # return 20 * np.log10(np.maximum(1e-5, x))
    return 20 * np.log10(np.maximum(1e-4, x))  # 最小为-80dB, 因为还有减去ref_dB


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _normalize(hp, S):
    # 这个做法存疑, 因为S>0时, 都会被截断成0, 即如果S>ref_db, 都会
    # 变成ref_db
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)


def _denormalize(hp, S):
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db


def _denormalize_tensorflow(hp, S):
    return (tf.clip_by_value(S, 0, 1) * -hp.min_level_db) + hp.min_level_db
