#!/bin/bash

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ ! -f path.sh ] || [ ! -f cmd.sh ]; then
    echo "Please change directory to e.g., egs/ljspeech/tts1"
    exit 1
fi

# shellcheck disable=SC1091
. ./path.sh || exit 1;
# shellcheck disable=SC1091
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=3
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
verbose=1      # verbose option

# feature configuration
fs=22050      # sampling frequency
fmax=7600     # maximum frequency
fmin=80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length


# dictionary related
dict=/home/rosen/Project/espnet/egs/blizzard13/tts2_gst/decode/new_gst_136/char_train_no_dev_units.txt
trans_type="char"

# embedding related
input_wav=

# decoding related
#synth_model=
synth_model=/home/rosen/Project/espnet/egs/blizzard13/tts2_gst/decode/new_gst_136/model.loss.best_newgst_e136
cmvn=/home/rosen/Project/espnet/egs/blizzard13/tts2_gst/decode/new_gst_136/cmvn_stft.ark
cmvn_mel=/home/rosen/Project/espnet/egs/blizzard13/tts2_gst/decode/new_gst_136/cmvn.ark

decode_config=/home/rosen/Project/espnet/egs/blizzard13/tts2_gst/decode/new_gst_136/decode.yaml
decode_dir=decode
griffin_lim_iters=64

# download related
models=ljspeech.transformer.v1
vocoder_models=ljspeech.parallel_wavegan.v1

#gst
ref_audio=CB-SCA-01-04
load_mel_decode=false
use_refaudio=true
token_weights=(3 0 0 0 0 0 0 0 0 0)
#token_weights=
ref_audio_dir=/home/rosen/Project/espnet/egs/blizzard13/tts2_gst/ref_audio
ref_audio=300000000


help_message=$(cat <<EOF
Usage:
    $ $0 <text>

Note:
    This code does not include text frontend part. Please clean the input
    text manually. Also, you need to modify feature configuration according
    to the model. Default setting is for ljspeech models, so if you want to
    use other pretrained models, please modify the parameters by yourself.
    For our provided models, you can find them in the tables at
    https://github.com/espnet/espnet#tts-demo.
    If you are beginner, instead of this script, I strongly recommend trying
    the following colab notebook at first, which includes all of the procedure
    from text frontend, feature generation, and waveform generation.
    https://colab.research.google.com/github/espnet/notebook/blob/master/tts_realtime_demo.ipynb

Example:
    # make text file and then generate it
    # (for the default model, ljspeech, we use upper-case char sequence as the input)
    echo "THIS IS A DEMONSTRATION OF TEXT TO SPEECH." > example.txt
    $0 example.txt

    # also you can use multiple text
    echo "THIS IS A DEMONSTRATION OF TEXT TO SPEECH." > example.txt
    echo "TEXT TO SPEECH IS A TECHQNIQUE TO CONVERT TEXT INTO SPEECH." >> example.txt
    $0 example.txt

    # you can specify the pretrained models
    $0 --models ljspeech.transformer.v3 example.txt

    # also you can specify vocoder model
    $0 --vocoder_models ljspeech.wavenet.mol.v2 example.txt

Available models:
    - ljspeech.tacotron2.v1
    - ljspeech.tacotron2.v2
    - ljspeech.tacotron2.v3
    - ljspeech.transformer.v1
    - ljspeech.transformer.v2
    - ljspeech.transformer.v3
    - ljspeech.fastspeech.v1
    - ljspeech.fastspeech.v2
    - ljspeech.fastspeech.v3
    - libritts.tacotron2.v1
    - libritts.transformer.v1
    - jsut.transformer.v1
    - jsut.tacotron2.v1
    - csmsc.transformer.v1
    - csmsc.fastspeech.v3

Available vocoder models:
    - ljspeech.wavenet.softmax.ns.v1
    - ljspeech.wavenet.mol.v1
    - ljspeech.parallel_wavegan.v1
    - libritts.wavenet.mol.v1
    - jsut.wavenet.mol.v1
    - jsut.parallel_wavegan.v1
    - csmsc.wavenet.mol.v1
    - csmsc.parallel_wavegan.v1
EOF
)

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

txt=$1
download_dir=${decode_dir}/download

if [ $# -ne 1 ]; then
    echo "${help_message}"
    exit 1;
fi

set -e
set -u
set -o pipefail


function download_models () {
    case "${models}" in
        "ljspeech.tacotron2.v1") share_url="https://drive.google.com/open?id=1dKzdaDpOkpx7kWZnvrvx2De7eZEdPHZs" ;;
        "ljspeech.tacotron2.v2") share_url="https://drive.google.com/open?id=11T9qw8rJlYzUdXvFjkjQjYrp3iGfQ15h" ;;
        "ljspeech.tacotron2.v3") share_url="https://drive.google.com/open?id=1hiZn14ITUDM1nkn-GkaN_M3oaTOUcn1n" ;;
        "ljspeech.transformer.v1") share_url="https://drive.google.com/open?id=13DR-RB5wrbMqBGx_MC655VZlsEq52DyS" ;;
        "ljspeech.transformer.v2") share_url="https://drive.google.com/open?id=1xxAwPuUph23RnlC5gym7qDM02ZCW9Unp" ;;
        "ljspeech.transformer.v3") share_url="https://drive.google.com/open?id=1M_w7nxI6AfbtSHpMO-exILnAc_aUYvXP" ;;
        "ljspeech.fastspeech.v1") share_url="https://drive.google.com/open?id=17RUNFLP4SSTbGA01xWRJo7RkR876xM0i" ;;
        "ljspeech.fastspeech.v2") share_url="https://drive.google.com/open?id=1zD-2GMrWM3thaDpS3h3rkTU4jIC0wc5B";;
        "ljspeech.fastspeech.v3") share_url="https://drive.google.com/open?id=1W86YEQ6KbuUTIvVURLqKtSNqe_eI2GDN";;
        "libritts.tacotron2.v1") share_url="https://drive.google.com/open?id=1iAXwC0AuWusa9AcFeUVkcNLG0I-hnSr3" ;;
        "libritts.transformer.v1") share_url="https://drive.google.com/open?id=1Xj73mDPuuPH8GsyNO8GnOC3mn0_OK4g3";;
        "jsut.transformer.v1") share_url="https://drive.google.com/open?id=1mEnZfBKqA4eT6Bn0eRZuP6lNzL-IL3VD" ;;
        "jsut.tacotron2.v1") share_url="https://drive.google.com/open?id=1kp5M4VvmagDmYckFJa78WGqh1drb_P9t" ;;
        "csmsc.transformer.v1") share_url="https://drive.google.com/open?id=1bTSygvonv5TS6-iuYsOIUWpN2atGnyhZ";;
        "csmsc.fastspeech.v3") share_url="https://drive.google.com/open?id=1T8thxkAxjGFPXPWPTcKLvHnd6lG0-82R";;
        *) echo "No such models: ${models}"; exit 1 ;;
    esac

    dir=${download_dir}/${models}
    mkdir -p "${dir}"
    if [ ! -e "${dir}/.complete" ]; then
        download_from_google_drive.sh "${share_url}" "${dir}" "tar.gz"
	touch "${dir}/.complete"
    fi
}

function download_vocoder_models () {
    case "${vocoder_models}" in
        "ljspeech.wavenet.softmax.ns.v1") share_url="https://drive.google.com/open?id=1eA1VcRS9jzFa-DovyTgJLQ_jmwOLIi8L";;
        "ljspeech.wavenet.mol.v1") share_url="https://drive.google.com/open?id=1sY7gEUg39QaO1szuN62-Llst9TrFno2t";;
        "ljspeech.parallel_wavegan.v1") share_url="https://drive.google.com/open?id=1tv9GKyRT4CDsvUWKwH3s_OfXkiTi0gw7";;
        "libritts.wavenet.mol.v1") share_url="https://drive.google.com/open?id=1jHUUmQFjWiQGyDd7ZeiCThSjjpbF_B4h";;
        "jsut.wavenet.mol.v1") share_url="https://drive.google.com/open?id=187xvyNbmJVZ0EZ1XHCdyjZHTXK9EcfkK";;
        "jsut.parallel_wavegan.v1") share_url="https://drive.google.com/open?id=1OwrUQzAmvjj1x9cDhnZPp6dqtsEqGEJM";;
        "csmsc.wavenet.mol.v1") share_url="https://drive.google.com/open?id=1PsjFRV5eUP0HHwBaRYya9smKy5ghXKzj";;
        "csmsc.parallel_wavegan.v1") share_url="https://drive.google.com/open?id=10M6H88jEUGbRWBmU1Ff2VaTmOAeL8CEy";;
        *) echo "No such models: ${vocoder_models}"; exit 1 ;;
    esac

    dir=${download_dir}/${vocoder_models}
    mkdir -p "${dir}"
    if [ ! -e "${dir}/.complete" ]; then
        download_from_google_drive.sh "${share_url}" "${dir}" ".tar.gz"
	touch "${dir}/.complete"
    fi
}

# Download trained models
if [ -z "${cmvn}" ]; then
    download_models
    cmvn=$(find "${download_dir}/${models}" -name "cmvn.ark" | head -n 1)
fi
if [ -z "${dict}" ]; then
    download_models
    dict=$(find "${download_dir}/${models}" -name "*_units.txt" | head -n 1)
fi
if [ -z "${synth_model}" ]; then
    download_models
    synth_model=$(find "${download_dir}/${models}" -name "model*.best" | head -n 1)
fi
if [ -z "${decode_config}" ]; then
    download_models
    decode_config=$(find "${download_dir}/${models}" -name "decode*.yaml" | head -n 1)
fi

synth_json=$(basename "${synth_model}")
model_json="$(dirname "${synth_model}")/${synth_json%%.*}.json"
use_speaker_embedding=$(grep use_speaker_embedding "${model_json}" | sed -e "s/.*: \(.*\),/\1/")


if [ "${use_speaker_embedding}" = "false" ] || [ "${use_speaker_embedding}" = "0" ]; then
    use_input_wav=false
else
    use_input_wav=true
fi



if [ -z "${input_wav}" ] && "${use_input_wav}"; then
    download_models
    input_wav=$(find "${download_dir}/${models}" -name "*.wav" | head -n 1)
fi

# Check file existence
if [ ! -f "${cmvn}" ]; then
    echo "No such CMVN file: ${cmvn}"
    exit 1
fi
if [ ! -f "${dict}" ]; then
    echo "No such dictionary: ${dict}"
    exit 1
fi
if [ ! -f "${synth_model}" ]; then
    echo "No such E2E model: ${synth_model}"
    exit 1
fi
if [ ! -f "${decode_config}" ]; then
    echo "No such config file: ${decode_config}"
    exit 1
fi
if [ ! -f "${input_wav}" ] && ${use_input_wav}; then
    echo "No such WAV file for extracting meta information: ${input_wav}"
    exit 1
fi
if [ ! -f "${txt}" ]; then
    echo "No such txt file: ${txt}"
    exit 1
fi

decode_base_dir=${decode_dir}
base=$(basename "${txt}" .txt)
ref_dir=${decode_dir}/ref
decode_dir=${decode_dir}/${base}
conf_dir=${decode_base_dir}/new_gst_136


if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "stage 0: Data preparation"

    # Create Text json
    [ -e "${decode_dir}/data" ] && rm -rf "${decode_dir}/data"
    mkdir -p "${decode_dir}/data"
    num_lines=$(wc -l < "${txt}")
    for idx in $(seq "${num_lines}"); do
        echo "${base}_${idx} X" >> "${decode_dir}/data/wav.scp"
        echo "X ${base}_${idx}" >> "${decode_dir}/data/spk2utt"
        echo "${base}_${idx} X" >> "${decode_dir}/data/utt2spk"
        echo -n "${base}_${idx} " >> "${decode_dir}/data/text"
        sed -n "${idx}"p "${txt}" >> "${decode_dir}/data/text"
    done
    mkdir -p "${decode_dir}/dump"
    data2json.sh --trans_type "${trans_type}" "${decode_dir}/data" "${dict}" > "${decode_dir}/dump/data.json"

    # Create ref_audio json
    [ -e "${ref_dir}/data" ] && rm -rf "${ref_dir}/data"
    mkdir -p "${ref_dir}/data"
    ref_audios=$(find "${ref_audio_dir}" -name *.wav | sort)
    for ref_a in ${ref_audios}; do
        ref_base=$(basename ${ref_a} .wav)
        echo "${ref_base} ${ref_a}"  >> "${ref_dir}/data/wav.scp"
        echo "${ref_base} X" >> "${ref_dir}/data/utt2spk"
        echo "${ref_base} N" >> "${ref_dir}/data/text"
    done
    utils/utt2spk_to_spk2utt.pl "${ref_dir}/data/utt2spk" > "${ref_dir}/data/spk2utt"
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ] && "${use_refaudio}"; then
    fbankdir=${ref_dir}/fbank
    [ ! -e ${fbankdir} ] && mkdir -p ${fbankdir}
    make_fbank.sh  \
        --fs ${fs} \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --win_length "${win_length}" \
        --n_mels ${n_mels} \
        ${ref_dir}/data \
        ${ref_dir}/log \
        ${fbankdir}

    #compute-cmvn-stats scp:${ref_dir}/data/feats.scp ${conf_dir}/cmvn.ark
    mkdir -p "${ref_dir}/dump"
    dump.sh --do_delta false ${ref_dir}/data/feats.scp ${cmvn_mel} ${ref_dir}/log ${ref_dir}/dump
    data2json.sh --feat "${ref_dir}/dump/feats.scp" --trans_type "${trans_type}" "${ref_dir}/data" "${dict}" > \
    "${ref_dir}/dump/data.json"

fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ] && "${use_input_wav}"; then
    echo "stage 1: x-vector extraction"

    utils/data/resample_data_dir.sh 22050

    utils/copy_data_dir.sh "${decode_dir}/data" "${decode_dir}/data2"
    sed -i -e "s;X$;${input_wav};g" "${decode_dir}/data2/wav.scp"
    utils/data/resample_data_dir.sh 16000 "${decode_dir}/data2"
    # shellcheck disable=SC2154
    steps/make_mfcc.sh \
        --write-utt2num-frames true \
        --mfcc-config conf/mfcc.conf \
        --nj 1 --cmd "${train_cmd}" \
        "${decode_dir}/data2" "${decode_dir}/log" "${decode_dir}/mfcc"
    utils/fix_data_dir.sh "${decode_dir}/data2"
    sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
        "${decode_dir}/data2" "${decode_dir}/log" "${decode_dir}/mfcc"
    utils/fix_data_dir.sh "${decode_dir}/data2"

    nnet_dir=${download_dir}/xvector_nnet_1a
    if [ ! -e "${nnet_dir}" ]; then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a "${download_dir}"
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi
    sid/nnet3/xvector/extract_xvectors.sh --cmd "${train_cmd} --mem 4G" --nj 1 \
        "${nnet_dir}" "${decode_dir}/data2" \
        "${decode_dir}/xvectors"

    local/update_json.sh "${decode_dir}/dump/data.json" "${decode_dir}/xvectors/xvector.scp"
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "stage 2: Decoding"
    # shellcheck disable=SC2154
    ${decode_cmd} "${decode_dir}/log/decode.log" \
        tts_decode.py \
        --config "${decode_config}" \
        --ngpu "${ngpu}" \
        --backend "${backend}" \
        --debugmode "${debugmode}" \
        --verbose "${verbose}" \
        --out "${decode_dir}/outputs_${ref_audio}/feats" \
        --json "${decode_dir}/dump/data.json" \
        --model "${synth_model}" \
        --use-refaudio "${use_refaudio}" \
        --ref-json "${ref_dir}/dump/data.json" \
        --load-mel-decode "${load_mel_decode}" \
        --ref-audio "${ref_audio}"
        #--token-weights "${token_weights}"
        #--ref-audio "${ref_audio}"
        #--token-weights ${token_weights}
        #--ref-audio "${ref_audio}" \
        #--token-weights "${token_weights}"
        #--use-refaudio "${use_refaudio}"
        #--token-weights "${token_weights}"
        #--ref-embed "${synth_model}" \
        #--ref-audio "${ref_audio}" \
fi

outdir=${decode_dir}/outputs_${ref_audio}; mkdir -p "${outdir}_denorm"
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "stage 3: Synthesis with Griffin-Lim"

    apply-cmvn --norm-vars=true --reverse=true "${cmvn}" \
        scp:"${outdir}/feats.scp" \
        ark,scp:"${outdir}_denorm/feats.ark,${outdir}_denorm/feats.scp"

    convert_fbank.sh --nj 1 --cmd "${decode_cmd}" \
        --fs "${fs}" \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft "${n_fft}" \
        --n_shift "${n_shift}" \
        --win_length "${win_length}" \
        --iters "${griffin_lim_iters}" \
        "${outdir}_denorm" \
        "${decode_dir}/log" \
        "${decode_dir}/wav_${ref_audio}"

    echo ""
    echo "Synthesized wav: ${decode_dir}/wav_${ref_audio}/${base}.wav"
    echo ""
    echo "Finished"
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "stage 4: Synthesis with Neural Vocoder"
    model_corpus=$(echo ${models} | cut -d. -f 1)
    vocoder_model_corpus=$(echo ${vocoder_models} | cut -d. -f 1)
    if [ "${model_corpus}" != "${vocoder_model_corpus}" ]; then
        echo "${vocoder_models} does not support ${models} (Due to the sampling rate mismatch)."
        exit 1
    fi
    download_vocoder_models
    dst_dir=${decode_dir}/wav_wnv_${ref_audio}

    # This is hardcoded for now.
    if [[ "${vocoder_models}" == *".mol."* ]]; then
        # Needs to use https://github.com/r9y9/wavenet_vocoder
        # that supports mixture of logistics/gaussians
        MDN_WAVENET_VOC_DIR=./local/r9y9_wavenet_vocoder
        if [ ! -d "${MDN_WAVENET_VOC_DIR}" ]; then
            git clone https://github.com/r9y9/wavenet_vocoder "${MDN_WAVENET_VOC_DIR}"
            cd "${MDN_WAVENET_VOC_DIR}" && pip install . && cd -
        fi
        checkpoint=$(find "${download_dir}/${vocoder_models}" -name "*.pth" | head -n 1)
        feats2npy.py "${outdir}/feats.scp" "${outdir}_npy"
        python3 ${MDN_WAVENET_VOC_DIR}/evaluate.py "${outdir}_npy" "${checkpoint}" "${dst_dir}" \
            --hparams "batch_size=1" \
            --verbose "${verbose}"
        rm -rf "${outdir}_npy"
    elif [[ "${vocoder_models}" == *".parallel_wavegan."* ]]; then
        checkpoint=$(find "${download_dir}/${vocoder_models}" -name "*.pkl" | head -n 1)
        if ! command -v parallel-wavegan-decode > /dev/null; then
            pip install parallel-wavegan
        fi
        parallel-wavegan-decode \
            --scp "${outdir}/feats.scp" \
            --checkpoint "${checkpoint}" \
            --outdir "${dst_dir}" \
            --verbose "${verbose}"
    else
        checkpoint=$(find "${download_dir}/${vocoder_models}" -name "checkpoint*" | head -n 1)
        generate_wav.sh --nj 1 --cmd "${decode_cmd}" \
            --fs "${fs}" \
            --n_fft "${n_fft}" \
            --n_shift "${n_shift}" \
            "${checkpoint}" \
            "${outdir}_denorm" \
            "${decode_dir}/log" \
            "${dst_dir}"
    fi
    echo ""
    echo "Synthesized wav: ${decode_dir}/wav_wnv_${ref_audio}/${base}.wav"
    echo ""
    echo "Finished"
fi
