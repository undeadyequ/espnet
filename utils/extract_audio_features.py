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

from sklearn.preprocessing import MinMaxScaler


def extract_feature(
        audio_f: str,
        sr: int = 44100,
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

    feature_list = []
    y = []
    if isinstance(audio_f, str):
        y, _ = librosa.load(audio_f, sr)
    elif isinstance(audio_f, np.ndarray):
        y = audio_f
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
    pitch, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch = [0 if math.isnan(p) else p for p in pitch]
    feature_list.append(np.mean(pitch))
    feature_list.append(np.std(pitch))

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