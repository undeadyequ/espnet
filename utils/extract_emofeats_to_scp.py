import pandas as pd
import argparse
from utils.extract_audio_features import extract_feature


def extract_emo(wav_scp, out_csv):
    pd_path = pd.read_csv(wav_scp)
    pd_emo = pd.DataFrame()
    for name, path in pd_path[0]:
        emo_feat = extract_feature(path)
        pd_emo.append(name, emo_feat)
    pd_emo.to_csv(out_csv)


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--scp", default="", type=str)
    parser.add_argument("--csv", default="emo_feats.csv", type=str)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    extract_emo(args.scp, args.csc)