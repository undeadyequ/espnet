import torch
from typing import Tuple, Dict, Any
import librosa
import numpy as np
import time
import os
import math
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract

from utils.textgrid import TextGrid, Tier

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
        #p3 = time.time()

        # auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
        pitch, _, _ = librosa.pyin(speech, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch = [0 if math.isnan(p) else p for p in pitch]

        #p4 = time.time()

        emofeats_list.append(np.mean(pitch))
        emofeats_list.append(np.std(pitch))

        return emofeats_list

    def _extract_pitch(self):




def extract_pitch(sig, sr=22050):
    pitch, _, _ = librosa.pyin(sig, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    pitch = [0 if math.isnan(p) else p for p in pitch]

    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)
    pitch_range = np.max(pitch) - np.min(pitch)

    return pitch_mean, pitch_std, pitch_range

def extract_energy(sig, sr=22050):
    pass

def extract_harmonic(sig, sr=22050):
    pass

def extract_silence(sig, sr=22050):
    pass

def extract_duration(sig, text, tools, type="phone", sr=22050):
    """
    extract duration of phone or words
    Args:
        sig:
        text:
        tools:
        type: phone or word
        sr:

    Returns:
    """
    # force alignment
    if tools == "montreal-forced-aligner":
        executor = "tools/force_alignment/montreal-forced-aligner/bin/mfa_align"
        wavtext_dir = "wav"
        lexicon = "tools/force_alignment/librispeech-lexicon.txt"
        lang = "english"
        outdir = "out"
        current_dir = os.path.dirname()

        executor = os.path.join(current_dir, executor)
        lexicon = os.path.join(current_dir, lexicon)
        cmd = "{} {} {} {} {}".format(executor, wavtext_dir, lexicon, lang, outdir)

        os.system(cmd)

    # duration for each
    duration = dict()
    for f in os.listdir(outdir):
        fid = TextGrid.load(f)
        for i, tier in enumerate(fid):
            if tier.nameid == type:
                for st, end, unit in tier.simple_transcript:
                    dur = float(end) - float(st)
                    if unit in duration.keys():
                        duration[unit].append(dur)
                    else:
                        duration[unit] = [dur]
            else:
                pass
    # duration statistic in word/phone
    duration_stats = dict()
    for k, v in duration.items():
        duration_stats[k] = (np.mean(v), np.std(v))

    # duration in utterance
    duration_utter_mean = 0
    for m, std in duration_stats.values():
        duration_utter_mean += m
    duration_utter_mean /= len(duration_stats)

    return duration, duration_stats, duration_utter_mean


def extract_spectral_tilt(sig, text):
    pass