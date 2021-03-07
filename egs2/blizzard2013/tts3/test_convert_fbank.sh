#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

convert_fbank.sh \
    --n_mels 80 \
    --n_shift 256 \
    --fmax 7600 \
    --fmin 80 \
    ../../../test_utils/spectrogram_data/ \
    ../../../test_utils/spectrogram_data/log \
    ../../../test_utils/spectrogram_data/wav