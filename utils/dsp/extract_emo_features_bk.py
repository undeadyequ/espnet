"""
This script extract features from existing audio vectors
"""

import pickle
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import math
import collections
import os

from sklearn.preprocessing import MinMaxScaler
import argparse
from utils.dsp.silenceremove import remove_silence_from_wav

def extract_emo_feature(
        audio: str,
        sr: int = 22050,
        normlaize: bool = False,
        remove_silence: bool = True,
        min_max_stats_f: str = "./normal/iemocap_train_feats_stats.csv"):
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
    feature_list = []
    y = []
    if isinstance(audio, str):
        if remove_silence:
            temp_f = "temp.wav"
            remove_silence_from_wav(audio, agress=2, out_wav=temp_f)
            y, _ = librosa.load(temp_f, sr)
            os.remove(temp_f)
        else:
            y, _ = librosa.load(audio, sr)
    elif isinstance(audio, np.ndarray):
        y = audio
    # 1. sig
    sig_mean = np.mean(abs(y))
    feature_list.append(sig_mean)  # sig_mean
    feature_list.append(np.std(y))  # sig_std

    # 2. rmse
    rmse = librosa.feature.rms(y + 0.0001)[0]
    feature_list.append(np.mean(rmse))  # rmse_mean
    feature_list.append(np.std(rmse))  # rmse_std

    # 3. silence
    silence = 0
    for e in rmse:
        if e <= 0.4 * np.mean(rmse):
            silence += 1
    silence /= float(len(rmse))
    feature_list.append(silence)  # silence

    # 4. harmonic
    y_harmonic = librosa.effects.hpss(y)[0]
    feature_list.append(np.mean(y_harmonic) * 1000)  # harmonic (scaled by 1000)

    # 5. pitch (instead of auto_correlation)
    cl = 0.45 * sig_mean
    center_clipped = []
    for s in y:
        if s >= cl:
            center_clipped.append(s - cl)
        elif s <= -cl:
            center_clipped.append(s + cl)
        elif np.abs(s) < cl:
            center_clipped.append(0)
    # auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
    pitch, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    pitch = [0 if math.isnan(p) else p for p in pitch]
    feature_list.append(np.mean(pitch))
    feature_list.append(np.std(pitch))
    # feature_list.append(1000 * np.max(auto_corrs)/len(auto_corrs))  # auto_corr_max (scaled by 1000)
    # feature_list.append(np.std(auto_corrs))  # auto_corr_std

    feature_list = np.array(feature_list).reshape(1, -1)

    # Check Normalization

    if normlaize:
        if min_max_stats_f is None:
            raise IOError("Must input min_max_stats_f")
        else:
            scalar = MinMaxScaler()
            train_feats_stats = pd.read_csv(min_max_stats_f)
            scalar.fit(train_feats_stats)
            feature_list = scalar.transform(feature_list)

    return feature_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str)
    parser.add_argument("--sr", default=22050)
    parser.add_argument("--normalize", default=True)
    parser.add_argument("--min_max_stats_f", default="/home/Data/blizzard2013_part_preprocess/dump/emo_feats/feats_stats.csv")

    args = parser.parse_args()
    feats_ls = extract_emo_feature(args.audio, args.sr, args.normalize, args.min_max_stats_f)
    print(feats_ls)