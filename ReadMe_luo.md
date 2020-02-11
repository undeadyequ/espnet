---
layout: post
title:  "espnet"
date:   2019-7-17 12:52:48 +0900
categories: jekyll update
---
### Good point
    - Varied config source
        - cmd config, env, Config file, default

    - Process flow control(bash script)
        - Stage execute, Control config
        - Model, method exchange by control config

    - Generalized Programming
        - Dynamic import from config
        - Specific parameter gatheredly set inside model (add_parameter, easily
            changed from outside)
        - TTS_interface
        - Generalized argment adding in all tts
        - Generalized train process in all tts

    - Raw data and features separation

    - Project architecture
        - experiment by data -> Model
        - Model(Include Parameter) is Generalized
        - Tools(Kaldi) is Generalized


### Project architecture
- espnet
  - tools
    - kaldi
  - espnet
    - bin
      - **tts_train.py** (Parameter settings)
      - lm_train.py
      - asr_train.py
    - tts
      - pytorch_backend
        - **tts.py**     (Train process)
    - nets (All dl model architecture)
      - pytorch_backend
        - tacotron2
          - cbhg.py
          - decoder.py
          - encoder.py
        - transformer
          - ...
        - **e2e_tts_tacotron2.py**
        - wavenet.py
    - transform(wav data transform)
        - spectrogram
  - egs (example by data)
    - ljspeech
      - tts1 (Algorithm different)
        - conf (Hyper-param file)
          - decode.yml
          - gpu.yml
          - ...
        - util
          -
        - local (5_step, )
          - data_download.sh
          - data_pre.sh
          -
        - run.sh (Hardware, Hyper-param, 5_Step(download, prepare, train, decode??, synthesis))
        - cmd.sh
        - path.sh
      - tts2
        - ?
    - blizzard
  - utils
    - dump.sh
    - convert_fbank.sh
  - tools(kaldi file)
    - reming

![architect](architect.png)

### Process flow

stage-1: Data Download

stage 0: Data preparation

    -
    local/data_prep.sh ${db_root}/LJSpeech-1.1 data/${trans_type}_train ${trans_type}
    utils/validate_data_dir.sh --no-feats data/${trans_type}_train
stage 1: Feature Generation

    - Generate the fbank features; by default 80-dimensional fbanks on each frame
    make_fbank.sh ...

    - make a dev set
    utils/subset_data_dir.sh --last data/${trans_type}_train 500 data/${trans_type}_deveval
    ..

    - compute statistics for global mean-variance normalization
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
    ...

    - dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${trans_type}_train ${feat_tr_dir}
    ...

    - Task dependent. You have to check non-linguistic symbols used in the corpus.

stage 2: Dictionary and Json Data Preparation

    - make json labels


stage 3: Text-to-speech model training

    - setup feature and duration for fastspeech knowledge distillation training
    local/setup_knowledge_dist.sh ...

    - Training
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

    tts_train.py -> tts.py -> e2e_tts_tacotron2.py -> encoder.py ...

stage 4: Decoding

    - average
    average_checkpoints.py

    - decode
    tts_decode.py

stage 5: Synthesis

    - convert_fbank
    convert_fbank.sh ..

stage 6: Objective Evaluation

    - evaluate cer
    local/ob_eval/evaluate_cer.sh
