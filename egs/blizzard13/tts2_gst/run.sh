#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
# [stage 6] 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=5
stop_stage=6
ngpu=1       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=8        # numebr of parallel jobs
dumpdir=/home/Data/program_data/espnet2/dump # directory to dump full features
featsdir=/home/Data/program_data/espnet2  # directory to save fbank and stft
verbose=1    # verbose option (if set > 0, get more log)
N=1000         # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
#resume="exp/char_train_no_dev_pytorch_train_pytorch_tacotron2+cbhg+gst/results/snapshot.ep.110"    # the snapshot path to resume (if set empty, no effect)
resume=""

# feature extraction related
fs=22050      # sampling frequency
fmax=7600     # maximum frequency
fmin=80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# char or phn
# In the case of phn, input transcription is convered to phoneem using https://github.com/Kyubyong/g2p.
trans_type="char"

# config files
#train_config=conf/train_pytorch_tacotron2+cbhg+local+gst.yaml # you can select from conf or conf/tuning.
train_config=conf/train_pytorch_tacotron2+cbhg+local+gst.yaml # you can select from conf or conf/tuning.

                                               # now we support tacotron2, transformer, and fastspeech
                                               # see more info in the header of each config.
decode_config=conf/decode.yaml


# decoding related
model=model.loss.best_newgst_e136
#model=snapshot.ep.99
n_average=0 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim


# root directory of db
db_root=/home/Data/blizzard2013/new_download

# exp tag
tag="" # tag for managing experiments.

# reference audio
ref_wav=/home/rosen/Desktop/CB-SCA-01-100.wav

#. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="${trans_type}_train_no_dev"
dev_set="${trans_type}_dev"
eval_set="${trans_type}_eval"


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    #local/data_download.sh ${db_root}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    #local/data_prep.sh ${db_root} data/${trans_type}_train ${trans_type} ${fs}
    utils/validate_data_dir.sh --no-feats --no-spk-sort data/${trans_type}_train
    # 2nd field is diff from 1st  (..10 ..100  VS ..10_abc ..100_abc)
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    # Generate the fbank features; by default 80-dimensional fbanks on each frame
    fbankdir=${featsdir}/fbank

    make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
        --fs ${fs} \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --win_length "${win_length}" \
        --n_mels ${n_mels} \
        data/${trans_type}_train \
        exp/${trans_type}_make_fbank/train \
        ${fbankdir}

    # make a dev set
    utils/subset_data_dir.sh --last data/${trans_type}_train 500 data/${trans_type}_deveval
    utils/subset_data_dir.sh --last data/${trans_type}_deveval 250 data/${eval_set}
    utils/subset_data_dir.sh --first data/${trans_type}_deveval 250 data/${dev_set}
    n=$(( $(wc -l < data/${trans_type}_train/wav.scp) - 500 ))
    utils/subset_data_dir.sh --first data/${trans_type}_train ${n} data/${train_set}

    # compute statistics for global mean-variance normalization
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${trans_type}_train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${trans_type}_dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${trans_type}_eval ${feat_ev_dir}
fi


dict=data/lang_1${trans_type}/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1${trans_type}/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 --trans_type ${trans_type} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type ${trans_type} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type ${trans_type} \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp --trans_type ${trans_type} \
         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Spectrogram extraction"
    stftdir=${featsdir}/stft
    for name in ${train_set} ${dev_set} ${eval_set}; do
        utils/copy_data_dir.sh data/${name} data/${name}_stft
        make_stft.sh --nj ${nj} --cmd "$train_cmd" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            data/${name}_stft \
            exp/make_stft/${name} \
            ${stftdir}

        utils/fix_data_dir.sh data/${name}_stft
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}_stft/feats.scp data/${train_set}_stft/cmvn.ark
    for name in ${train_set} ${dev_set} ${eval_set}; do
        # dump features for training
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            data/${name}_stft/feats.scp \
            data/${train_set}_stft/cmvn.ark \
            exp/dump_feats/${name}_stft \
            ${dumpdir}/${name}_stft
        # update json
        local/update_json.sh ${dumpdir}/${name}/data.json \
            ${dumpdir}/${name}_stft/feats.scp
    done
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model training"
    tr_json=${feat_tr_dir}/data.json
    dt_json=${feat_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

if [ ${n_average} -gt 0 ]; then
    model=model.last${n_average}.avg.best
fi
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})_sad

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [ ${n_average} -gt 0 ]; then
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${model} \
                               --num ${n_average}
                               --num ${n_average}
    fi
    pids=() # initialize pids
    #ref_id="willa_cather_CB-SCA-01-135" # sad
    #ref_id="willa_cather_CB-SCA-01-168" # json
    #ref_json=${outdir}/${dev_set}/data.json
    #ref_weight=${weight_json}
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        #cp ${dumpdir}/${name}/data.json ${outdir}/${name}
        cp ${dumpdir}/${name}/data_small.json ${outdir}/${name}/data.json
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis"
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}_stft/cmvn.ark \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp
        convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs} \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --iters ${griffin_lim_iters} \
            ${outdir}_denorm/${name} \
            ${outdir}_denorm/${name}/log \
            ${outdir}_denorm/${name}/wav
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: style decode+synthesis in one"
    pids=() # initialize pids
fi