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


## Error
utils/combine_data.sh [info]: not combining utt2uniq as it does not exist
utils/combine_data.sh [info]: not combining segments as it does not exist
utils/combine_data.sh: combined utt2spk
utils/combine_data.sh [info]: not combining utt2lang as it does not exist
utils/combine_data.sh [info]: not combining utt2dur as it does not exist
utils/combine_data.sh [info]: not combining utt2num_frames as it does not exist
utils/combine_data.sh [info]: not combining reco2dur as it does not exist
utils/combine_data.sh [info]: not combining feats.scp as it does not exist
utils/combine_data.sh: combined text
utils/combine_data.sh [info]: not combining cmvn.scp as it does not exist
utils/combine_data.sh [info]: not combining vad.scp as it does not exist
utils/combine_data.sh [info]: not combining reco2file_and_channel as it does not exist
utils/combine_data.sh: combined wav.scp
utils/combine_data.sh [info]: not combining spk2gender as it does not exist
utils/fix_data_dir.sh: file /home/Data/blizzard2013_part_preprocess/data/tr_no_dev/spk2utt is not in sorted order or not unique, sorting it
- /home/Data/blizzard2013_part_preprocess/data/tr_no_dev/utt2spk differ: char 322587, line 7266
utt2spk is not in sorted order when sorted first on speaker-id 
(fix this by making speaker-ids prefixes of utt-ids)


## cut_data.sh

cat utt2spk | grep charlotte_bronte_CB > utt2spk_part
cat text | grep charlotte_bronte_CB > text_part
cat wav.scp | grep charlotte_bronte_CB > wav_part.scp
cat utt2num_frames | grep charlotte_bronte_CB > utt2num_frames_part
cat feats.scp | grep charlotte_bronte_CB > feats_part.scp
cat spk2utt | grep charlotte_bronte_CB > spk2utt_part


# start env

source ../../../../tools/venv/bin/activate espnet
