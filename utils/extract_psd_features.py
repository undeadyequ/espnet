"""
This script extract features from existing audio vectors
"""
import librosa
import numpy as np
import time
import os
import math

from utils.textgrid import TextGrid, Tier
from matplotlib import pyplot as plt



def extract_psd_feature(
        audio: str,
        text_f: str = None,
        sr: int = 22050,
        normlaize: bool = False,
        min_max_stats_f: str = None):
    """
    extract feature like below:
    sig:
    rmse:
    silence:
    harmonic:
    pitch:

    audio: audio file or audio list
    return feature_list: np of [n_samples, n_features]
    """
    y = []
    if isinstance(audio, str):
        y, _ = librosa.load(audio, sr)
    elif isinstance(audio, np.ndarray):
        y = audio

    # Create folder including wav and text files
    temp_in_dir = "temp_in"
    temp_out_dir = "temp_out"
    os.system("mkdir {}".format(temp_in_dir))
    os.system("cp {} {}").format(audio, temp_in_dir)
    os.system("cp {} {}").format(text_f, temp_in_dir)
    # phone duration
    phone = extract_duration(temp_in_dir, temp_out_dir)

    # pitch pitch range
    pitch, _, pitch_range = extract_pitch(y, sr)

    # energy
    eng = extract_energy(y, sr)


    # spectral tilt
    spc_tilt = extract_spectral_tilt(y)

    return [pitch, pitch_range, phone, eng, spc_tilt]




def extract_sig(y, sr=22050):
    sig_mean = np.mean(abs(y))
    return sig_mean, np.std(y)


def extract_pitch(sig, sr=22050):
    # pitch
    pitch, _, _ = librosa.pyin(sig, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    pitch = [0 if math.isnan(p) else p for p in pitch]

    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)
    pitch_range = np.max(pitch) - np.min(pitch)

    return pitch_mean, pitch_std, pitch_range


def extract_energy(sig, sr=22050):
    # rmse
    rmse = librosa.feature.rms(sig + 0.0001)[0]
    return np.mean(rmse), np.std(rmse)


def extract_harmonic(sig, sr=22050):
    # harmonic
    y_harmonic = librosa.effects.hpss(sig)[0]
    np.mean(y_harmonic) * 1000                  # harmonic (scaled by 1000)
    return y_harmonic


def extract_silence(sig, sr=22050):
    # silence
    rmse = librosa.feature.rms(sig + 0.0001)[0]
    silence = 0
    for e in rmse:
        if e <= 0.4 * np.mean(rmse):
            silence += 1
    silence /= float(len(rmse))
    return silence


def force_alignment(executor="tools/force_alignment/montreal-forced-aligner/bin/mfa_align",
                    wavtext_dir="wav",
                    lexicon="tools/force_alignment/librispeech-lexicon.txt",
                    lang="english",
                    outdir="out"):
    """
    use montreal-forced-aligner
    Args:
        executor:
        wavtext_dir:
        lexicon:
        lang:
        outdir:

    Returns:

    """
    cmd = "{} {} {} {} {}".format(executor, wavtext_dir, lexicon, lang, outdir)
    os.system(cmd)


def extract_duration(wavtext_dir,
                     out_dir,
                     tools="montreal-forced-aligner",
                     type="phone",
                     sr=22050):
    """
    extract duration of phone or words
    Args:
        sig:
        text:
        tools:
        type: phone or word
        sr:

    Returns:
    """
    # force alignment
    if tools == "montreal-forced-aligner":
        force_alignment(wavtext_dir, out_dir)

    # parser
    # duration for each phone
    duration = dict()
    for f in os.listdir(out_dir):
        fid = TextGrid.load(f)
        for i, tier in enumerate(fid):
            if tier.nameid == type:
                for st, end, unit in tier.simple_transcript:
                    dur = float(end) - float(st)
                    if unit in duration.keys():
                        duration[unit].append(dur)
                    else:
                        duration[unit] = [dur]
            else:
                pass
    # duration statistic in word/phone
    duration_stats = dict()
    for k, v in duration.items():
        duration_stats[k] = (np.mean(v), np.std(v))

    # duration in utterance
    duration_utter_mean = 0
    for m, std in duration_stats.values():
        duration_utter_mean += m
    duration_utter_mean /= len(duration_stats)

    return duration, duration_stats, duration_utter_mean


def extract_spectral_tilt(sig):
    """
    predictor coefficient of first order all pole model, followed the controllable tts paper

    Args:
        sig:
        text:

    Returns:

    """
    a = librosa.lpc(sig, 1)[1]
    return a


if __name__ == '__main__':
    executor = "tools/force_alignment/montreal-forced-aligner/bin/mfa_align"
    wavtext_dir = "wav"
    lexicon = "tools/force_alignment/librispeech-lexicon.txt"
    lang = "english"
    outdir = "out"
    extract_duration()