---
layout: post
title:  "espnet"
date:   2019-7-17 12:52:48 +0900
categories: jekyll update
---
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

### Process flow
![architect](architect.png)

## Order
./run.sh -> tts_train.py -> tts.py -> e2e_tts_tacotron2.py

## key code

```python
# network architecture
¥# tts_train.py
parser.add_argument('--model-module', type=str, default="espnet.nets.pytorch_backend.e2e_tts_tacotron2:Tacotron2",
                    help='model defined module')

¥# Dynamic load Netwarks by model name
model_class = dynamic_import(train_args.model_module)
model = model_class(idim, odim, train_args)
assert isinstance(model, TTSInterface)
logging.info(model)

¥# e2e_tts_tacotron2.py
class Tacotron2(TTSInterface, torch.nn.Module):
  @staticmethod
  def add_arguments(args):
    """Add model-specific arguments to the parser."""
    group = parser.add_argument_group("tacotron2 model setting")")
    # encoder
    group.add_argument('--embed-dim', default=512, type=int,
                       help='Number of dimension of embedding')
    ...
  def __init__(self, idim, odim, args):
    args = fill_missing_args(args, self.add_arguments)

¥# dynamic_import.py
import importlib
def dynamic_import(import_path, alias=dict()):
  model_path, model_name = import_path.split(":")
  importlib(model_path)

```
