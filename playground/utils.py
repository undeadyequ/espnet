"""
DSP related
fpath <-> wav <-> mels <-> mags
"""
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import librosa
from Hyperparameters import Hyperparameters as hp
import os
import copy
def get_spectrograms(fpath):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''

    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)
    spec = copy.deepcopy(linear)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel_ori = np.dot(mel_basis, mag)  # (n_mels, t)
    mel_ori = mel_ori.T.astype(np.float32)  # (T, n_mels)


    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel_ori))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel_ori, mag, spec


def load_spectrograms(fpath):
    """
    mels : (Num_frame, n_mels * r)
    """
    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0  # for reduction
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    return fname, mel.reshape((-1, hp.n_mels * hp.r)), mag



def mel2wav(mel):
    """
    mel: (T, n_mels)

    """
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    inv_mel_basis = np.linalg.pinv(mel_basis)   # (1 + n_fft/2, n_mels)
    mag = np.dot(inv_mel_basis, mel.T).T    # (T, 1+n_fft/2)
    return spectrogram2wav(mag)


def spectrogram2wav(mag):
    """ (T_y, fft//2 + 1) """
    # Transpose
    mag = mag.T

    # Denormalize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav construction
    wav, spec_syn = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav, spec_syn

def save_att_png(att_weight, png_f):
    """
    att_weight : (T_y/r ,T_x)
    """
    plt.imshow(att_weight.T, cmap='hot', interpolation="nearest")
    plt.xlabel("Decoder Steps")
    plt.ylabel("Encoder Steps")
    plt.savefig(png_f, format="png")

def griffin_lim(spectrogram):
    """
    spectrogram : [1 + fft_n / 2, t]
    """
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    #y = np.real(X_t)
    return X_t, X_best

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")




if __name__ == '__main__':
    ### Test save_att_png
    att = np.random.randn(3, 2)
    png = "att.png"
    #save_att_png(att, png)

    #
