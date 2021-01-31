
import librosa
from scipy import io
import scipy
import numpy as np
import matplotlib.pyplot as plt
import soundfile

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


if __name__ == '__main__':
    main()