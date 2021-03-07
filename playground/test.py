
import librosa
from scipy import io
import scipy
import numpy as np
import matplotlib.pyplot as plt
import soundfile
from playground.utils import spectrogram2wav, get_spectrograms, mel2wav

from scipy.io.wavfile import write

def main():
    y, sr = librosa.load("/home/rosen/Project/espnet/ref_audio/emotion/CB-SCA-01-135_sad.wav", duration=0.020)
    soundfile.write("trumpet.wav", librosa.load(librosa.ex('trumpet'))[0], sr)
    a = librosa.lpc(y, 5)
    print(a[1])
    b = np.hstack([[0], -1 * a[1:]])
    y_hat = scipy.signal.lfilter(b, [1], y)
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.plot(y_hat, linestyle='--')
    ax.legend(['y', 'y_hat'])
    ax.set_title('LP Model Forward Prediction')

    plt.savefig("out1.png")


def test_griffin_lim():
    wav_f = "/home/rosen/Project/espnet/playground/CB-SCA-01-135_sad.wav"
    y, sr = librosa.load(wav_f)
    #mels = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=256, n_mels=80)

    mels, mags, spec = get_spectrograms(wav_f)
    mels = mels.T

    print(mels.shape)

    assert mels.shape[0] == 80
    fmin = 0
    fmax = sr / 2
    #mspc = np.power(10.0, mels)

    EPS = 1e-10
    mel_basis = librosa.filters.mel(sr, n_fft=1024, n_mels=80, fmin=fmin, fmax=fmax)

    inv_mel_basis = np.linalg.pinv(mel_basis)
    spc = np.maximum(EPS, np.dot(inv_mel_basis, mels).T)


    # vocoder

    # use librosa's fast Grriffin-Lim algorithm
    #spc = np.abs(spc.T)

    """
    y = librosa.griffinlim(
        S=spc,
        n_iter=0,
        hop_length=256,
        win_length=1024,
        window="hann",
        center=True if spc.shape[1] > 1 else False,
    )
    """

    y, _ = spectrogram2wav(spc)

    soundfile.write("out_2.wav", y, sr, "PCM_16")


def wav2mel2wav(wav_f, waf_cv_f):
    mels, mags, spec = get_spectrograms(wav_f)
    wav_cv, spec_syn = mel2wav(mels)
    write(waf_cv_f, 22050, wav_cv)

def wav2spec2wav(wav_f, waf_cv_f):
    mels, mags, spec = get_spectrograms(wav_f)
    wav_cv, spec_syn = spectrogram2wav(mags)
    write(waf_cv_f, 22050, wav_cv)


if __name__ == '__main__':
    #test_griffin_lim()
    wav_f = "CB-SCA-01-135_sad.wav"
    wav2mel2wav(wav_f, "out_4.wav")