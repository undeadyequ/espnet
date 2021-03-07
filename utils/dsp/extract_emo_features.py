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


def extract_emo_feature_modify(
        audio: str,
        sr: int = 22050,
        remove_silence: bool = True,
        normalize: bool = False,
        min_max_stats_f: str = "./normal/iemocap_train_feats_stats.csv"):
    """
    All features all extracted under silence-free
    extract feature like below:
    rmse:
    pitch:
    harmonic:
    pitch:
    normalize: if normalize needed

    audio: audio file or audio list
    return feature_list: np of [n_samples, n_features]
    """

    feature_list = []
    y = []
    if isinstance(audio, str):
        if remove_silence:
            temp_f = "temp.wav"
            remove_silence_from_wav(audio, agress=1, out_wav=temp_f)
            y, _ = librosa.load(temp_f, sr)
            #os.remove(temp_f)
        else:
            y, _ = librosa.load(audio, sr)

    # 2. rmse
    rmse = librosa.feature.rms(y + 0.0001)[0]
    lg_rmse = [20 * math.log(r) / math.log(10) for r in rmse]
    feature_list.append(np.mean(lg_rmse))  # rmse_mean
    feature_list.append(np.std(lg_rmse))  # rmse_std
    feature_list.append(np.max(lg_rmse) - np.min(lg_rmse))  # rmse_range

    # 3. harmonic
    y_harmonic = librosa.effects.hpss(y)[0]
    feature_list.append(np.mean(y_harmonic) * 1000 )  # harmonic (scaled by 1000)
    feature_list.append(np.std(y_harmonic) * 1000 )  # harmonic (scaled by 1000)

    # 4. pitch (instead of auto_correlation)
    pitch, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    lg_pitch_nzero = [math.log(p) / math.log(10) for p in pitch if not math.isnan(p) and p != 0]
    if len(lg_pitch_nzero) == 0:
        lg_pitch_nzero = [0]
    feature_list.append(np.mean(lg_pitch_nzero))
    feature_list.append(np.std(lg_pitch_nzero))
    feature_list.append(np.max(lg_pitch_nzero) - np.min(lg_pitch_nzero))

    feature_list = np.array(feature_list).reshape(1, -1)

    # Check Normalization
    if normalize:
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
    feats_ls = extract_emo_feature_modify(args.audio, args.sr, args.normalize, args.min_max_stats_f)
    print(feats_ls)