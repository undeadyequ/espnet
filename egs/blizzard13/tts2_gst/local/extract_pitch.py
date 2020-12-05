import matplotlib.pylab as plt
import librosa
import numpy as np
import os


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def extract_f0(audio):
    """

    :param audio: string
    :return:
    f0 : [shape=(frame_n:)]
    """
    y, sr = librosa.load(audio)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return f0


def extract_energy(audio):
    y, sr = librosa.load(audio)
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)
    return np.squeeze(rms)


def speech_rate(audio):
    pass


def show_pitch(audios):
    """

    :param audios: [shape=(audio_n:)]
    :return:
    """

    pic_out = "out_pitch.png"
    fig, ax = plt.subplots()
    cmap = get_cmap(len(audios)*2)
    for i, a in enumerate(audios):
        f0 = extract_f0(a)
        times = np.array(range(len(f0)))
        lable = os.path.basename(a)[:-4]
        ax.set(title="Pitch in all tokens")
        ax.plot(times, f0, label=lable, color=cmap(i*2), linewidth=1)
        ax.legend(loc='upper right')
        plt.savefig(pic_out)


def show_energy(audios):
    pic_out = "out_energy.png"
    fig, ax = plt.subplots()
    cmap = get_cmap(len(audios)*2)
    for i, a in enumerate(audios):
        eny = extract_energy(a)
        times = np.array(range(len(eny)))
        #times = librosa.times_like(eny)
        lable = os.path.basename(a)[:-4]
        ax.set(title="Energy in all tokens")
        ax.plot(times, eny, label=lable, color=cmap(i*2), linewidth=1)
        ax.legend(loc='upper right')
        plt.savefig(pic_out)



def test():
    y, sr = librosa.load(librosa.ex('trumpet'))
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots()
    #img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='pYIN fundamental frequency estimation')
    #fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.plot(times, f0-100, label='f1', color='blue', linewidth=3)

    ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    wav_tokens_dir = "/home/rosen/Project/espnet/egs/blizzard13/tts2_gst/decode/syn_list/wav_tokens"
    wav_list = os.listdir(wav_tokens_dir)
    wav_list = [os.path.join(wav_tokens_dir, w) for w in wav_list]
    show_pitch(wav_list)
    show_energy(wav_list)