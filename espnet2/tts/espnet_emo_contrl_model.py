from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import Optional
from typing import Tuple


import torch
from typeguard import check_argument_types

from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.abs_emotts import AbsEmoTTS
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.emofeats_extract import Emofeats_extract


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class EspnetEmoTTSModel(AbsESPnetModel):
    def __init__(
        self,
        mel_extract: Optional,
        emo_feats_extract: Optional,
        pretrained_SER: Optional,
        tts: AbsEmoTTS
    ):
        self.mel_extract = mel_extract
        self.emo_feats_extract = emo_feats_extract
        self.tts = tts
        self.pretrained_ser = pretrained_SER
        super().__init__()

    def forward(self,
                text: torch.Tensor,
                text_lengths: torch.Tensor,
                speech: torch.Tensor,
                speech_lengths: torch.Tensor
                ):
        # Extract emotional feature
        emo_feats = self.emo_feats_extract(speech)
        mels = self.mel_extract(speech)

        # Compute emo distributions here because pretrained_ser is not trainable
        emo_distrb = self.pretrained_ser(emo_feats)

        return self.tts(
                text=text,
                speech=mels,
                emo_distrbs=emo_distrb)

    def inference(self,
                  text: torch.Tensor,
                  text_lengths: torch.Tensor,
                  speech: torch.Tensor,
                  speech_lengths: torch.Tensor,
                  emo_feats: torch.Tensor,
                  emo_distrbs: torch.Tensor,
                  ):
        """
        3 types inference
            1. with reference audio
            2. with emotional distributions
            3. with emotional features
        """
        if speech is not None:
            emo_feats = self.emo_feats_extract(speech)
            emo_distrbs = self.pretrained_ser(emo_feats)
        elif emo_feats is not  None:
            emo_distrbs = self.pretrained_ser(emo_feats)
        elif emo_distrbs is not None:
            pass
        else:
            raise IOError

        return self.tts.inference(
            text=text,
            emo_distrbs=emo_distrbs
        )