"""
extract feature from id_wav_f.
"id", "rmse_m", "rmse_std",
             "rmse_range", "harm_m", "harm_std",
             "pitch_m", "pitch_std", "pitch_range"
wav_file,label,rmse_m,rmse_std,rmse_range,harm_m,harm_std,pitch_m,pitch_std,pitch_range

"""

import pandas as pd
import argparse
from utils.dsp.extract_emo_features import extract_emo_feature_modify
import time
from utils.classify_emo_from_feats import cls_emo_pretrained_ml
from joblib import dump, load
from espnet2.tts.feats_extract.prosody_feats_extract import ProsodyFeatsExtract
import librosa
from sklearn.preprocessing import MinMaxScaler


def extract_emofeats_from_scp(
        wav_scp: str,
        out_csv: str,
        normal_stats_f: str = "",
        clf_f: str = "",
        fs: int=22050,
        silence_remove=False
):
    """
    Extract emofeats, lab, distrbs at one time
    Args:
        wav_scp:
        out_csv:
        normal_stats_f:
        clf_f:

    Returns:

    """
    pd_path = pd.read_csv(wav_scp, delim_whitespace=True, header=None)
    pd_emo = pd.DataFrame()
    emo_feats = []
    normal = False
    if normal_stats_f != "":
        normal = True
    if clf_f != "":
        clf = load(clf_f)
    for _, row in pd_path.iterrows():
        st = time.time()
        audio_id = row[0]
        audio_path = row[1]
        #y, _ = librosa.load(audio_path, fs)  # including downsampling
        audio_emo_feat = extract_emo_feature_modify(audio_path, sr=fs, normalize=normal,
                                                    remove_silence=silence_remove, min_max_stats_f=normal_stats_f)
        print(audio_emo_feat)

        row = [audio_id]
        row.extend(audio_emo_feat[0, :])
        """
        if clf_f != "":
            emo_res, emo_prob = cls_emo_pretrained_ml(clf, audio_emo_feat)
            row.append(emo_res)
            row.extend(emo_prob)
        """
        emo_feats.append(row)
        ed = time.time()
        print("duration time:{}".format((ed - st)))
    pd_emo = pd_emo.append(emo_feats)
    pd_emo.to_csv(out_csv, sep=" ", header=False, index=False)


def normalize_audio_fts(
        in_csv,
        method : str = None,
        normalizer_stats : str = None,
        out_stats_csv="maxminstate.csv",
        out_csv="audio_feats_stats.csv3",
        ):
    """
    Normalize audio fts by Maxmin scaler
    Args:
        in_csv:
        method:
            if maxmin  :
            if quantile:
        out_csv:
        normalizer_stats:
            When Method is not None, This is not activated.
            Normalized by MaxMin scaler
        out_stats_csv: Out put current max min scaler


    Returns:
    """
    QUANTILE_MIN = 0.2
    QUANTILE_MAX = 0.8

    df = pd.read_csv(in_csv, header=None, sep=" ")
    if df.shape[1] == 9:
        df.columns = ["id", "rmse_m", "rmse_std",
             "rmse_range", "harm_m", "harm_std",
             "pitch_m", "pitch_std", "pitch_range"]
    print("before norm:\n", df.describe())

    if method == "quantile":
        normalizer_stats = None
        q_20, q_80 = df.quantile(QUANTILE_MIN), df.quantile(QUANTILE_MAX)
        for col in df.iloc[:, 1:]:
            df.loc[df[col] > q_80[col], col] = q_80[col]
            df.loc[df[col] < q_20[col], col] = q_20[col]
        print("after quantile:\n", df.describe())

        scalar = MinMaxScaler()
        df[df.columns[1:]] = scalar.fit_transform(df[df.columns[1:]])
        # Save feats stats
        feats_stats = pd.DataFrame()
        feats_stats["min"] = scalar.data_min_
        feats_stats["max"] = scalar.data_max_
        feats_stats = feats_stats.transpose()
        feats_stats.to_csv(out_stats_csv, index=False)

    if normalizer_stats is not None:
        scalar = MinMaxScaler()
        train_feats_stats = pd.read_csv(normalizer_stats)
        print("MaxMin Sclaer:\n", train_feats_stats)
        scalar.fit(train_feats_stats)
        df[df.columns[1:]] = scalar.transform(df[df.columns[1:]])

    df.to_csv(out_csv, header=None, sep=" ", index=False)


def get_id_wpath_pair(wav_scp):
    """
    Args:
        wav_scp:

    Returns:
        id_wpath
        [(id1, wpath1),(id2, wpath2), ...]

    """
    pd_path = pd.read_csv(wav_scp, delim_whitespace=True, header=None)
    id_wpath = []
    for _, row in pd_path.iterrows():
        id_wpath.append((row[0], row[1]))
    return id_wpath


def extract_psd_feats(wav_scp):
    """

    Args:
        wav_scp:

    Returns:

    """
    pass



def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--scp", default="/home/Data/blizzard2013_part_preprocess/dump/fbank/tr_no_dev/wav_test.scp", type=str)
    parser.add_argument("--out_csv", default="emo_feats_tr_no_dev_test.csv", type=str)
    parser.add_argument("--norm_f", default="/home/Data/blizzard2013_part_preprocess/dump/emo_feats/feats_stats.csv", type=str)
    parser.add_argument("--clf_f", default="/home/Data/blizzard2013_part_preprocess/dump/emo_feats/rf_classifier.pkl",
                        type=str)
    parser.add_argument("--fs", default=22050,
                        type=int)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    extract_emofeats_from_scp(args.scp, args.out_csv, args.norm_f, args.clf_f, args.fs)