---
layout: post
title:  "espnet"
date:   2019-7-17 12:52:48 +0900
categories: jekyll update
---

### Pretrained model list

![list](https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv)

### Tabel

    - TTS model
    - Data structure
    - Project architecture
    - Training/Decoding/Synthesis flow

### TTS models in EPSnet
    - Tacotron
    - Taoctron2
    - Transformer
    - Fastspeech


## Main Function

### Data Preprocess

    - tts.decode()
data.json ->{make_batchset} List[List[Tuple[str, dict]]]...[0]
->{LoadInputsAndTargets(one_utt...)} (List[ndarray], List[ndarray], ...[1])
->{model.inference} outs, probs, att_wights  ...[2]


...[0]:
1st list: Batch List(Bath size equals gpu_numbers)
2nd list: Utterance List


...[1]:

...[2]:(in tts)
output is values of Middle Value, showed below:
Middle Value: OrderedDict([(x_name, List[ndarray]), (y_name, List[ndarray], (spembs_name, List[ndarray]), (spcs_name, List[ndarray]))])...
x_name: target1
y_name: input1
spcs_name: input2

    - tts.train()
data.json ->{make_batchset} List[List[Tuple[str, dict]]]
->{lambda} List[Tuple[str, dict]]
->{LoadInputsAndTargets} (List[ndarray], List[ndarray], ...)
->{CustomConverter} {"xs": List[tensor], "ilens":List[tensor],
                     "ys": List[tensor], "labels":, List[tensor],
                     "olens:": List[tensor]}...[1]
->{TransformDataset} dataset...
->{ChainerDataLoader} train_iter
->{CustomUpdater} updater
->{training.Trainer} trainer
->{trainer.extend(CustomEvaluater)}
->{trainer.extend(torch_snapshot())}
->{trainer.extend(snapshot_object(model, "model.loss.best"))}
->{trainer.extend(att_reporter)}
->{trainer.extend(extensions.PlotReport(plot_key))
->{trainer.extend(extensions.LogReport)
->{trainer.extend(extensions.PrintReport)
->{trainer.extend(extensions.ProgressBar())
->{trainer.extend(TensorboardLogger())
->{trainer.extend(ShufflingEnabler([train_iter]))
->{trainer.run()}
->{check_early_stop(trainer, args.epochs)}


...[1]
"labels" is for stop prediction
""
...[1]
CustomConverter and batchset are capsulized in ChainerDataLoader


### data.json structure

 - data_json
data_json = {
        "utt1": {
          "utt2spk": "Tom",
          "input": [
                          {
                              "feat": "/Users/rosen/speech_project/tts/espnet/egs/blizzard13/tts2_gst/dump/char_train_no_dev/feats.1.ark:29",
                              "name": "input1",
                              "shape": [
                                  968,
                                  80]
                          },
                          {
                              "feat": "/Users/rosen/speech_project/tts/espnet/egs/blizzard13/tts2_gst/dump/char_train_no_dev/feats.4.ark:29",
                              "name": "input2",
                              "shape": [
                                  968,
                                  513]
                          }
                      ],
          "output": [
                    {
                    "name": "target1",
                    "shape": [108,42],
                    "text": "JANE EYRE AN AUTOBIOGRAPHY BY CHARLOTTE BRONTE CHAPTER I THERE WAS NO POSSIBILITY OF TAKING A WALK THAT DAY.",
                    "token": "J A N E <space> E Y R E <space> A N <space> A U T O B I O G R A P H Y <space> B Y <space> C H A R L O T T E <space> B R O N T E <space> C H A P T E R <space> I <space> T H E R E <space> W A S <space> N O <space> P O S S I B I L I T Y <space> O F <space> T A K I N G <space> A <space> W A L K <space> T H A T <space> D A Y .",
                    "tokenid": "19 10 23 14 8 14 34 27 14 8 10 23 8 10 30 29 24 11 18 24 16 27 10 25 17 34 8 11 34 8 12 17 10 27 21 24 29 29 14 8 11 27 24 23 29 14 8 12 17 10 25 29 14 27 8 18 8 29 17 14 27 14 8 32 10 28 8 23 24 8 25 24 28 28 18 11 18 21 18 29 34 8 24 15 8 29 10 20 18 23 16 8 10 8 32 10 21 20 8 29 17 10 29 8 13 10 34 6"
                    }]
                  }
      },
      {
        "utt2": {
        ...
      },
        ...
      }
  - Make




## Experiment

1. train experiment
    - Real train
        exp/char_train_no_dev_pytorch_train_pytorch_tacotron2+cbhg+gst
    - Debug train


2. Decode/Synthesis experiment
    - real decode/synthesis
        model.loss.best_decode         # decode
        model.loss.best_decode_denorm  # synthesis

    - debug decode
        ...decode_ys_inference        # delete style branch
        ...decode_ys_inference_denorm #

1. ck_style_concnt

   --backend pytorch --ngpu 1 --minibatches 0 --outdir ../../egs/blizzard13/tts2_gst/exp/blizzard2013_char_train_no_dev_pytorch_train_pytorch_tacotron2+cbhg+gst/results_ck_style_concnt --tensorboard-dir ../../egs/blizzard13/tts2_gst/tensorboard/blizzard2013_char_train_no_dev_pytorch_train_pytorch_tacotron2+cbhg+gst/ck_style_concnt --verbose 1 --seed 1 --train-json  /home/Data/program_data/espnet2/dump/char_train_no_dev/data.json --valid-json /home/Data/program_data/espnet2/dump/char_dev/data.json --config ../../egs/blizzard13/tts2_gst/conf/train_pytorch_tacotron2+cbhg+gst.yaml

2.


## Install
- install system level package, cmake, sox, sndfile, ffmpeg, flac

- Install Kaldi and set soft link inside of espnet
    - ln -s ../../tts/espnet/tools/kaldi
- Install Espnet
    - Set cuda
        - ./setup_cuda_env.sh /usr/local/cuda
    - set Conda env
        - ./setup_anaconda.sh ./venv
        - install numpy, configargparse
    - Make
        - cd ../tools
        - make   # install python library
        -



## Energy, pitch, harmonic extraction
    - Extract feature token-wise or word-wise
    - Duration: Tokenized duration in each phoneme(or word)
        - Example (call)
            0:0:23 ~ 0:0:26 : KY
            0:0:26 ~ 0:0:29 : Al

    - Speech_lenght: Extract specific lenght


## Algorithm
    ## Fastspeech
        - Teacherforce of Energy and Pitch in inference

    ## Fastspeech2
    -

## Arguments process
    - append Args by Class.add_parameter(parser)
    - Args inputs should be matched with appended Args
    - Class including TTSTask, add all args in class_choices_list(used espnetModel), and AbsTask

##

## Mel-bank computing
   - Firstly comput stft from wav by librosa.stft
   - then multiply mel with log to compute mel_bank
   - make_fbank.sh => compute-fbank-feats.py => logmelspectrogram => librosa.stft

## Segmentation info of audio
    - Used in vctk

##

## Python skill
    - Argments Type

## waiting to do

    - emo_extract normalization


## Good implement

    Class Text2Speech( \**model_path ):
        @torch.no_grad()
        def  __call__(text: String, ref_wav: array-like(s_len,)):
            Pass
        @property
        def  use_speech() -> bool

## tts.sh process stages
    1. Data preparation
    2. Extract feature or raw
    3. Remove long/short sentence ( min/max frame_num)
    4. Feature in air (type = raw)
        -
    5. TTS collect statsStage TTS collect stats:
        => main_function/collect_stats(model, train_iter, valid_iter, ...) => espmodel.collect_stats

    6. Train


## mel-bank extraction

Firstly compute stft from wav by librosa.stft, then multiply mel with log to compute mel_bank
make_fbank.sh => compute-fbank-feats.py => logmelspectrogram => librosa.stft
ABCMeta? ABC? (used in AbsFeatsExtract?)
speech_lengths used in feature extraction? clip?



Class AbsTask(ABC):

## abstractTASK process
    -
def main_worker():
    """
    1. random-seed
    2. build model <= from task specific task, TTSTask etc.
    3. optimizer
    4. build scheduler
    5. Dump args to config.yaml
    6. Loads pre-trained model
    7. Resume the training state from the previous epoch
    8A. collect_stats
    8B. Build iterator factories
    9B. Start training


## abstractTASK VS ttsTASK
ttsTask build task-specific model
abstractTask use task-specific model with generate function(forward, inference)


## batch generation
    - Generate same seed by same epoch for reproductivity
    - include SequenceIterFactory

        iter_factory = cls.build_iter_factory(...)
        for epoch in range(1, max_epoch):
            for keys, batch in iter_factory.build_iter(epoch):
                model(\**batch)

        >>> iter_factory = cls.build_iter_factory(...)
        >>> for epoch in range(1, max_epoch):
        ...     for keys, batch in iter_fatory.build_iter(epoch):
        ...         model(\**batch)

## ESPnetDataset DataLoader
  - Example
  ```python

  class Mydataset(Dataset):
    def __init(self):
      self.mels
      self.labels
    def __getitem(self, index):
      mel =self.mels[index]
      lab = self.label[index]
      sample = {"mel": mel, "label":label}
      return sample
    def __len__(self):
      return len(self.labels)

  dataset = Mydataset()
  dataloader = DataLoader(dataset, collate_fn, batch_size=4, shuffle=True, num_worker=4)

  for i, samples in enumerate(dataloader):
    print(samples["mels"].size())
    print(samples["label"].size()

  DataLoader(
    dataset=self.dataset,
    batch_sampler=batches,
    num_workers=self.num_workers,
    pin_memory=self.pin_memory,
    **kwargs,)
  ```

  1. random-seed
  2. build model <= from task specific task, TTSTask etc.
  3. optimizer
  4. build scheduler
  5. Dump args to config.yaml
  6. Loads pre-trained model
  7. Resume the training state from the previous epoch
  8A. collect_stats
  8B. Build iterator factories
  9B. Start training


[AbsTask] ->main():None []
->main_worker(args):None [ESPnetTTSModel]
  ->build_model(args):model [TTSTask]
    # feats_extract
    ->feats_extractor_choices.get_class(args.feats_extract):feats_extract_class
    ->feats_extract_class(**args.feats_extract_conf):feats_extract
    # emo_extract
    ...
    ->emofeats_extract_class():emo_extract
    ...
    ->emo_classifier_class():emo_classifier
    # Normalizer
    ...
    # TTS
    ...
    ->tts_class(idim, odim, **args.tts_conf):tts
    # Extract component
    ...
    ->pitch_extract_class(**args.pitch_extract_conf):pitch_extract

    # Conclude
    ->ESPnetTTSModel(feats_extract, emo_extract, emo_classifier, tts):model

->main(cmd): [TTSTask=>AbsTask]
  ->main_worker(args): [AbsTask]
    ->build_optimizer(args, model=model):
    ->scheduler_classes.get(name):scheduler
    ->schedulers.append(scheduler)
    ->build_iter_factory(args):train_iter_factory []
      ->build_sequence_iter_factory(args, iter_option):SequenceIterFactory [] \
        ->ESPnetDataset(iter_option.data_path_and_name_and_type, iter_option.preprocess):dataset [torch.utils.data]
        ->build_batch_sampler(iter_option.batch_type, iter_option.shape_files, batch_size):batch_sampler [samplers]
          ->NumElementsBatchSampler()
        ->SequenceIterFactory(dataset, batches, args.seeds):seqFactor []
          ->self.build_iter(epoch): Dataloader []
    ->build_iter_factory(args):valid_iter_factory []
    ->trainer.run(model,
                  optimizers,
                  schedulers,
                  train_iter_factory,
                  valid_iter_factory,
                  seeds)
      ->
      ->train_one_epoch(model, train)
        -> model(**batch)

## args
  - Arguments Input
    - train.yaml
      - tts_conf is used to give args to model
      - optim_conf is used to ... optimizer
    - cmd args in run.sh and tts.sh

  - Parse the Arguments
    - add_arguments in AbsTask, TTSTask, trainer respectively
    - no need add_arguments in model since they were gathered by model_conf
    - implement:
    ->main():None [AbsTask]
      ->get_parser():config_argparser.ArgumentParser [ArgumentParser]
        - cls.add_task_arguments(parser): None [TTSTask]
        - cls.trainer.add_arguments(parser): None [Trainer]
      ->parser.parse_args(cmd): args  <= parse cmd, including train.yaml which furthermore parsed by ArgumentParser


  ```python
    dataset = ESPnetDataset(
        iter_options.data_path_and_name_and_type,
        float_dtype=args.train_dtype,
        preprocess=iter_options.preprocess_fn,
        max_cache_size=iter_options.max_cache_size,
        max_cache_fd=iter_options.max_cache_fd,
    )
    for iiter, (\_, batch) in enumerate(
        reporter.measure_iter_time(iterator, "iter_time"), 1
    ):
    - Iterator, used in trainer, is the a dataloader with indices comming from self-defined sampler

    train_iter_factory = cls.build_iter_factory(
         args=args,
         distributed_option=distributed_option,
         mode="train",
     )
  ```


## debug
--use_preprocessor true --token_type phn --token_list /home/Data/blizzard2013_part_preprocess/data/token_list/phn_tacotron_g2p_en_no_space/tokens.txt --non_linguistic_symbols none --cleaner tacotron --g2p g2p_en_no_space --normalize global_mvn --normalize_conf stats_file=/home/Data/blizzard2013_part_preprocess/exp/tts_stats_fbank_phn_tacotron_g2p_en_no_space/train/feats_stats.npz --resume false --fold_length 150 --fold_length 800 --output_dir /home/Data/blizzard2013_part_preprocess/exp/tts_train_fbank_phn_tacotron_g2p_en_no_space --config /home/rosen/Project/espnet/egs2/blizzard2013/tts3/conf/train.yaml --odim=80 --train_data_path_and_name_and_type /home/Data/blizzard2013_part_preprocess/dump/fbank/tr_no_dev/text,text,text --train_data_path_and_name_and_type
/home/Data/blizzard2013_part_preprocess/dump/emo_feats/emo_feats_etfs.csv,emo_feats,csv_float --train_data_path_and_name_and_type /home/Data/blizzard2013_part_preprocess/dump/fbank/tr_no_dev/feats.scp,speech,kaldi_ark --train_shape_file /home/Data/blizzard2013_part_preprocess/exp/tts_stats_fbank_phn_tacotron_g2p_en_no_space/train/text_shape.phn --train_shape_file /home/Data/blizzard2013_part_preprocess/exp/tts_stats_fbank_phn_tacotron_g2p_en_no_space/train/speech_shape --valid_data_path_and_name_and_type /home/Data/blizzard2013_part_preprocess/dump/fbank/dev/text,text,text --valid_data_path_and_name_and_type /home/Data/blizzard2013_part_preprocess/dump/fbank/dev/feats.scp,speech,kaldi_ark --valid_shape_file /home/Data/blizzard2013_part_preprocess/exp/tts_stats_fbank_phn_tacotron_g2p_en_no_space/valid/text_shape.phn --valid_shape_file /home/Data/blizzard2013_part_preprocess/exp/tts_stats_fbank_phn_tacotron_g2p_en_no_space/valid/speech_shape

## Kaldiio
  - used in file io directly from scp/ark to numpy array


## Dump ‘s subdirectory
  - fbank, stft, raw


## Question
speech audio,
speech txt same Length



l1_loss=1.185, mse_loss=1.162, bce_loss=0.197, psd_loss=0.009, attn_loss=0.015, loss=2.567

- 1/22
