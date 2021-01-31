import pandas as pd
import os
import argparse


def split_feats_lab_distb(fld_f):
    basic_dir = os.path.dirname(fld_f)
    basic_name = os.path.basename(fld_f)

    efts_f = os.path.join(basic_dir, basic_name[:-4] + "_etfs.csv")
    elbs_f = os.path.join(basic_dir, basic_name[:-4] + "_elbs.csv")
    edst_f = os.path.join(basic_dir, basic_name[:-4] + "_edst.csv")

    efts, elbs, edts = [], [], []

    with open(fld_f, "r") as infile:
        for ln in infile:
            row = ln.rstrip().split(" ")
            eft = row[0] + " " + ",".join(row[1:9])
            elb = row[0] + " " + row[9]
            edt = row[0] + " " + ",".join(row[10:15])
            efts.append(eft)
            elbs.append(elb)
            edts.append(edt)

    with open(efts_f, "w") as outfile:
        for item in efts:
            outfile.write(item + "\n")

    with open(elbs_f, "w") as outfile:
        for item in elbs:
            outfile.write(item + "\n")

    with open(edst_f, "w") as outfile:
        for item in edts:
            outfile.write(item + "\n")


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--fld_csv", default="/home/rosen/Project/espnet/files/emo_feats_eval.csv", type=str)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    split_feats_lab_distb(args.fld_csv)