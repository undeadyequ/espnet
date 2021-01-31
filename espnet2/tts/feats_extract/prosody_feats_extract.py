import torch
from typing import Tuple, Dict, Any
import librosa
import numpy as np
import time
import os
import math
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
import soundfile
import scipy
from utils.textgrid import TextGrid, Tier
from matplotlib import pyplot as plt

class ProsodyFeatsExtract(AbsFeatsExtract):
    """
    Feats used in SER:
        pitch(m,std), energy(m,std), harmonic, silence
    Feats used in Prosody control:
        pitch(m,range), phone duration, energy, spectral tilt
    """
    def __init__(self):
        super().__init__()

    def get_parameters(self) -> Dict[str, Any]:
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            center=self.stft.center,
            normalized=self.stft.normalized,
            use_token_averaged_energy=self.use_token_averaged_energy,
            reduction_factor=self.reduction_factor,
        )

    def output_size(self) -> int:
        raise 1

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        return emofeats_list