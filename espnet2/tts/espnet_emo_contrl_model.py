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
        emo_feats_normalize: Optional,
        pretrained_SER: Optional,
        tts: AbsEmoTTS
    ):
        super().__init__()
        self.mel_extract = mel_extract
        self.emo_feats_extract = emo_feats_extract
        self.tts = tts
        self.pretrained_ser = pretrained_SER

    def forward(self,
                text: torch.Tensor,
                text_lengths: torch.Tensor,
                speech: torch.Tensor,
                speech_lengths: torch.Tensor,
                emo_feats: torch.Tensor,
                emo_feats_lengths: torch.Tensor
                ):
        """
        :param text:
        :param text_lengths:
        :param speech:
        :param speech_lengths:
        :param emo_feats: (1, 8)
        :param emo_feats_length: (1)
        :return:
        """
        """
        # Extract emotional feature
        emo_feats = self.emo_feats_extract(speech, speech_lengths)
        mels = self.mel_extract(speech)

        # Compute emo distributions here because pretrained_ser is not trainable
        emo_distrb = self.pretrained_ser(emo_feats)
        """

        return self.tts(
            text=text,
            text_lengths=text_lengths,
            speech=speech,
            speech_lengths=speech_lengths,
            emo_feats=emo_feats,
            emo_feats_lengths=emo_feats_lengths
        )

    def collect_feats(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            emo_feats: torch.Tensor,
            emo_feats_lengths: torch.Tensor):
        """
        if self.mel_extract is not None:
            feats, feats_lengths = self.mel_extract(speech, speech_lengths)
        else:
            feats, feats_lengths = speech, speech_lengths

        if self.emo_feats_extract is not None:
            emo_feats = self.emo_feats_extract(speech)
            feats_dict.update(emo_feats=emo_feats)
        """
        feats_dict = {"feats": speech, "feats_lengths": speech_lengths}
        return feats_dict

    def inference(self,
                  text: torch.Tensor,
                  speech: torch.Tensor = None,
                  emo_feats: torch.Tensor = None,
                  emolabs: torch.Tensor = None,
                  spembs: torch.Tensor = None,
                  durations: torch.Tensor = None,
                  pitch: torch.Tensor = None,
                  energy: torch.Tensor = None,
                  **decode_config,
                  ):
        """
        3 types inference
            1. with reference audio
            2. with emotional distributions
            3. with emotional features    torch.Tensor(feats_dim)
        """

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
        """
        outs, probs, att_ws = self.tts.inference(
            text=text,
            speech=speech,
            emo_feats=emo_feats,
            emolabs=emolabs)
        """
        if self.normalize is not None:
            # NOTE: normalize.inverse is in-place operation
            outs_denorm = self.normalize.inverse(outs.clone()[None])[0][0]
        else:        
        """
        outs_denorm = outs
        return outs, outs_denorm, probs, att_ws