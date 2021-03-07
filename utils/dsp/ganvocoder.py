from parallel_wavegan.utils import load_model
import torch
import argparse
from scipy.io.wavfile import write
import time

VOCODER = "/home/rosen/Project/espnet/downloads/en/parallel_wavegan/ljspeech.parallel_wavegan.v2/checkpoint-400000steps.pkl"


def vocoder(mels, fs=22050, wav_f="out.wav", vocoder_path=None):
    if vocoder_path is None:
        vocoder_path = VOCODER
    vocoder = load_model(vocoder_path)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to("cpu")

    with torch.no_grad():
        start = time.time()
        y = vocoder.inference(mels)
    rtf = (time.time() - start) / (len(y) / fs)
    print("RTF = {}".format(rtf))
    write(wav_f, fs, y.view(-1).cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()