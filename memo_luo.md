### Implement
./run.sh --backend pytorch --ngpu 0

### gitignore file
```
// General

// recipes
egs/*/*/data*
egs/*/*/db
egs/*/*/downloads
egs/*/*/dump
egs/*/*/enhan
egs/*/*/exp
egs/*/*/fbank
egs/*/*/mfcc
egs/*/*/stft
egs/*/*/tensorboard
egs/*/*/wav*


// tools
tools/bin
tools/include
tools/lib
tools/lib64
tools/bats-core
tools/chainer_ctc/
tools/kaldi*
tools/miniconda.sh
tools/moses/
tools/mwerSegmenter/
tools/nkf/
tools/venv/
tools/sentencepiece/
tools/swig/
tools/warp-ctc/
tools/warp-transducer/
tools/*.done
tools/PESQ*
tools/hts_engine_API*
tools/open_jtalk*
tools/pyopenjtalk*
```

### other
  - model set
  ```python
    --model-module
  ```
  - Batch Size
    32 -> 16


### Config setting
```python
parser = get_parser()
args, _ = parser.parse_known_args(cmd_args)

from espnet.utils.dynamic_import import dynamic_import
model_class = dynamic_import(args.model_module)
assert issubclass(model_class, TTSInterface)
model_class.add_arguments(parser)
args = parser.parse_args(cmd_args)      # cmd_args including may model args
```

### Loging setting
  - Set inside python
  - Set in run.pl
```python
if args.verbose > 0:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
else:
    logging.basicConfig(
        level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    logging.warning('Skip DEBUG/INFO messages')
logging.warning("CUDA_VISIBLE_DEVICES is not set.")

```

### Configuration usage
```python
parser.add("--conf_v")   # values in configuration file
args, _ = parser.parse_known_args(cmd_args)
args.format_help()       # args usage
args.format_values()     # args values showing
```
