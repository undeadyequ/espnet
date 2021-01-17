## model: emocontrl_tts

## train command
.run.sh --dump /home/... --train_set --valid_set --test_sets

./run.sh --preprocess_dir /home/Data/blizzard2013_part_preprocess  --stage 6 --stop_stage 6 --feats_type fbank
./run.sh  --stage 6 --stop_stage 6 --feats_type fbank

batch_num = 500
train_time/batch = 1s
train_time/epoch = 1 * 500s = 8.3min


## evaluate command
1. evaluate directly
   
2. evaluate in real time

3. evaluate with provided pre-trained model in model_zoo

source ../tools/venv/bin/activate espnet
python syns.py  (real time)
python syns2.py (directly)


# 

## cut_data.sh

cat utt2spk | grep charlotte_bronte_CB > utt2spk_part
cat text | grep charlotte_bronte_CB > text_part
cat wav.scp | grep charlotte_bronte_CB > wav_part.scp
cat utt2num_frames | grep charlotte_bronte_CB > utt2num_frames_part
cat feats.scp | grep charlotte_bronte_CB > feats_part.scp
cat spk2utt | grep charlotte_bronte_CB > spk2utt_part


# start env

source ../../../../tools/venv/bin/activate espnet
