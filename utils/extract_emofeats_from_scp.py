import pandas as pd
import argparse
from utils.extract_audio_features import extract_feature
import time
from utils.classify_emo_from_feats import cls_emo_pretrained_ml
from joblib import dump, load

def extract_emofeats_from_scp(
        wav_scp: str,
        out_csv: str,
        normal_stats_f: str = "",
        clf_f: str = ""):

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
        audio_emo_feat = extract_feature(audio_path, sr=22050, normlaize=normal, min_max_stats_f=normal_stats_f)
        row = [audio_id]
        row.extend(audio_emo_feat[0, :])
        if clf_f != "":
            emo_res, emo_prob = cls_emo_pretrained_ml(clf, audio_emo_feat)
            row.append(emo_res)
            row.extend(emo_prob)
        emo_feats.append(row)
        ed = time.time()
        print("duration time:{}".format((ed - st)))
    pd_emo = pd_emo.append(emo_feats)
    pd_emo.to_csv(out_csv, sep=" ", header=False, index=False)



def extract_emofeats_lab_from_scp(
        wav_scp: str,
        out_csv: str,
        normal_stats_f: str = "",
        clf: str= ""
):
    extract_emofeats_from_scp(wav_scp, out_csv, normal_stats_f)


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--scp", default="/home/Data/blizzard2013_part_preprocess/dump/fbank/tr_no_dev/wav_test.scp", type=str)
    parser.add_argument("--csv", default="emo_feats_tr_no_dev_test.csv", type=str)
    parser.add_argument("--norm_f", default="/home/Data/blizzard2013_part_preprocess/dump/emo_feats/feats_stats.csv", type=str)
    parser.add_argument("--clf_f", default="/home/Data/blizzard2013_part_preprocess/dump/emo_feats/rf_classifier.pkl",
                        type=str)

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    extract_emofeats_from_scp(args.scp, args.csv, args.norm_f, args.clf_f)