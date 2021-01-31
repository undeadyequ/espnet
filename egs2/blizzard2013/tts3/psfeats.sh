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

scp_f="/home/Data/blizzard2013_part_preprocess/dump/fbank/tr_no_dev/wav_test.scp"
csv="/home/Data/blizzard2013_part_preprocess/dump/fbank/tr_no_dev/emo_feats_test.csv"
norm_f="/home/Data/blizzard2013_part_preprocess/dump/emo_feats/feats_stats.csv"
clf_f="/home/Data/blizzard2013_part_preprocess/dump/emo_feats/rf_classifier.pkl"

#
${python} -m utils.extract_emofeats_from_scp
    --scp "${scp_f}" \
    --csv "${csv}" \
    --norm_f "${norm_f}" \
    --clf_f "${clf_f}"

${python} -m utils.split_emofeats_lab_distrb
    --fld_csv "${csv}"