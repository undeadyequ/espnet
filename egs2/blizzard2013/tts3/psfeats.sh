#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh
. ./cmd.sh


python=python3

scp_f="/home/Data/blizzard2013_part_preprocess/dump/fbank/tr_no_dev/wav.scp"
csv="/home/Data/blizzard2013_part_preprocess/dump/fbank/tr_no_dev/emo_feats.csv"
norm_f="/home/Data/blizzard2013_part_preprocess/dump/emo_feats/feats_stats.csv"
clf_f="/home/Data/blizzard2013_part_preprocess/dump/emo_feats/rf_classifier.pkl"

pkl=""
#
${python} -m utils.extract_emofeats \
    --scp "${scp_f}" \
    --out_csv "${csv}" \
#    --norm_f "${norm_f}" \
#    --clf_f "${clf_f}"

#${python} -m utils.extract_emofeats \
#    --txt "{txt_f}" \
#    --csv "" \
#    --pkl ""



#${python} -m utils.split_emofeats_lab_distrb
#    --fld_csv "${csv}"