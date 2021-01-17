import torch
from typing import Tuple, Dict, Any
import librosa
import numpy as np
import time
import math
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract


class Emofeats_extract(AbsFeatsExtract):
    """
    Extract emofeats: 10
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
        emofeats_list = []

        # sig
        sig_mean = np.mean(abs(speech))
        emofeats_list.append(sig_mean)  # sig_mean
        emofeats_list.append(np.std(sig_mean))  # sig_std

        # rmse
        rmse = librosa.feature.rms(speech + 0.0001)[0]
        emofeats_list.append(np.mean(rmse))  # rmse_mean
        emofeats_list.append(np.std(rmse))  # rmse_std

        # silence
        silence = 0
        for e in rmse:
            if e <= 0.4 * np.mean(rmse):
                silence += 1
        silence /= float(len(rmse))
        emofeats_list.append(silence)  # silence

        # harmonic
        y_harmonic = librosa.effects.hpss(speech)[0]
        emofeats_list.append(np.mean(y_harmonic) * 1000)  # harmonic (scaled by 1000)

        # pitch
        # http://access.feld.cvut.cz/view.php?cisloclanku=2009060001
        cl = 0.45 * sig_mean
        center_clipped = []
        for s in speech:
            if s >= cl:
                center_clipped.append(s - cl)
            elif s <= -cl:
                center_clipped.append(s + cl)
            elif np.abs(s) < cl:
                center_clipped.append(0)
        p3 = time.time()
        # auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
        pitch, _, _ = librosa.pyin(speech, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch = [0 if math.isnan(p) else p for p in pitch]
        p4 = time.time()
        print("audio size: {}, pitch:{}".format(len(y) / 44100.0, (p4 - p3)))

        emofeats_list.append(np.mean(pitch))
        emofeats_list.append(np.std(pitch))

        return emofeats_list