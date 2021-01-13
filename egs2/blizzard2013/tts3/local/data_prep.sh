#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

num_dev=5
num_eval=5
train_set="tr_no_dev"
dev_set="dev"
eval_set="eval1"
preprocess_dir="/home/Data/blizzard2013_part_preprocess"

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 [Options] <db>"
    echo "e.g.: $0 downloads/VCTK-Corpus"
    echo ""
    echo "Options:"
    echo "    --num_dev: number of development uttreances (default=${num_dev})."
    echo "    --num_eval: number of evaluation uttreances (default=${num_eval})."
    echo "    --train_set: name of train set (default=${train_set})."
    echo "    --dev_set: name of dev set (default=${dev_set})."
    echo "    --eval_set: name of eval set (default=${eval_set})."
    exit 1
fi

set -euo pipefail

# NOTE(kan-bayashi): p315 will not be used since it lacks txt data
spks=$(find "${db}/wav1" -maxdepth 1 -exec basename {} \; | sort | grep -v wav1)
train_data_dirs=""
dev_data_dirs=""
eval_data_dirs=""
for spk in ${spks}; do
    data=${preprocess_dir}/data
    [ ! -e ${data}/${spk}_train ] && mkdir -p ${data}/${spk}_train

    # set filenames
    scp=${data}/${spk}_train/wav.scp
    utt2spk=${data}/${spk}_train/utt2spk
    text=${data}/${spk}_train/text
    spk2utt=${data}/${spk}_train/spk2utt

    # check file existence
    [ -e "${scp}" ] && rm "${scp}"
    [ -e "${utt2spk}" ] && rm "${utt2spk}"
    [ -e "${text}" ] && rm "${text}"

    # make scp, text
    find "${db}/wav1/${spk}" -follow -name "*.wav" | sort | while read -r wav; do
        id=$(basename "${wav}" | sed -e "s/\.[^\.]*$//g")
        txt=${db}/txt1/${spk}/${id}.txt

        if [ ! -e "${txt}" ]; then
            echo "${id} does not have a text file. skipped."
            continue
        fi

        echo "${id} ${wav}" >> "${scp}"
        echo "${id} ${spk}" >> "${utt2spk}"
        echo "${id} $(cat ${txt})" >> "${text}"

        utils/utt2spk_to_spk2utt.pl "${utt2spk}" > "${spk2utt}"
    done

    # split
    num_all=$(wc -l < "${scp}")
    num_deveval=$((num_dev + num_eval))
    num_train=$((num_all - num_deveval))
    utils/subset_data_dir.sh --last "${data}/${spk}_train" "${num_deveval}" "${data}/${spk}_deveval"
    utils/subset_data_dir.sh --first "${data}/${spk}_deveval" "${num_dev}" "${data}/${spk}_${eval_set}"
    utils/subset_data_dir.sh --last "${data}/${spk}_deveval" "${num_eval}" "${data}/${spk}_${dev_set}"
    utils/subset_data_dir.sh --first "${data}/${spk}_train" "${num_train}" "${data}/${spk}_${train_set}"

    # remove tmp directories
    rm -rf "${data}/${spk}_train"
    rm -rf "${data}/${spk}_deveval"

    train_data_dirs+=" ${data}/${spk}_${train_set}"
    dev_data_dirs+=" ${data}/${spk}_${dev_set}"
    eval_data_dirs+=" ${data}/${spk}_${eval_set}"
done

utils/combine_data.sh ${data}/${train_set} ${train_data_dirs}
utils/combine_data.sh ${data}/${dev_set} ${dev_data_dirs}
utils/combine_data.sh ${data}/${eval_set} ${eval_data_dirs}

# remove tmp directories
rm -rf data/p[0-9]*

echo "Successfully prepared data."
