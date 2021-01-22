import pandas as pd
import os

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


if __name__ == '__main__':
    fld_f = "/home/rosen/Project/espnet/files/emo_feats_eval.csv"
    split_feats_lab_distb(fld_f)