#!/usr/bin/env bash

# prepare index file
db=$1

create_index=false
extrct_emofts=true
extrct_txtfts=true
extrct_combfts=true

# create index
if [[ ${db}=="iemocap" ]]; then
    db_dir=/home/Data/IEMOCAP_session_only
    id_wav_f=../index/iemocap/wav.csv
    id_emo_f=../index/iemocap/emo.csv
    id_text_f=../index/iemocap/text.csv
    clf_f=
    fs=16000
fi

#if [ ${create_index} ]; then
#    python preprocess/iemocap/data_prepare.py --db_dir ${db_dir} --id_wav_f ${id_wav_f} --id_emo_f ${id_emo_f} \
#                                              --id_text_f ${id_text_f}
#fi


# extract emo_fts and txt_fts
if [ ${extrct_emofts} ]; then
    python ../../utils/extract_emofeats.py --scp ${id_wav_f} \
                                         --out_csv emofeats.csv \
                                         --fs ${fs}
    # normalize
    python ../../utils/normalize emofeats.csv emofeats_norm.csv
fi

if [ ${extrct_emofts} ]; then
    python ../../utils/extract_txtfeats.py --scp ${id_wav_f} \
                                         --out_csv emofeats.csv \
                                         --fs ${fs}
fi


if [ ${extrct_combfts} ]; then
    python ../../utils/extract.py --scp ${id_wav_f} \
                                         --out_csv emofeats.csv \
                                         --fs ${fs}
fi

