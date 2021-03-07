import sys
sys.path.append("..")
sys.path.append(".")
from Synthesis import *
from emoTTS.nets import Hyperparameters as H
import os
import torch

syn_text = "it took me a little long time to learning speech generation . now that i have to go back work again."
model_path_300 = "../exp_result/contrl_gst/state/epoch_300.pt.pt"
model_path_200 = "../exp_result/contrl_gst/state/epoch_200.pt.pt"


def test_token_selection():
    # 1 for each token  <- Really bad
    for i in range(H.nums_token + 1):
        # Token scale
        token_scale_dir = os.path.join("syn_audio", "token_scale_test")
        if not os.path.exists(token_scale_dir):
            os.mkdir(token_scale_dir)
        scale = (0.8, 0.3)
        for s in scale:
            token_init = torch.tensor([
                0.0,  # 1
                0.0,  # 2
                0.0,  # 3
                0.0,  # 4
                0.0,  # 5
                0.0,  # 6
                0.0,  # 7
                0,  # 8
                0,  # 9
                0.0])  # 10
            if i != 0:
                token_init[i-1] = s
            #audio_f = os.path.join(token_scale_dir, "t_token_{}_scale_{}_model_{}.wav".format(i, s, 300))
            #syn_style_from_token(syn_text, model_path_300, audio_f, token_init)
            audio_f = os.path.join(token_scale_dir, "t_token_{}_scale_{}_model_{}.wav".format(i, s, 200))
            syn_style_from_token(syn_text, model_path_200, audio_f, token_init)
        # Token sample (Mixed diff Token)


def test_reference_audio():
    ref_audio = "../ref_wav/style_audio_angry.wav"
    audio_f = os.path.join("syn_audio", "t_ref_1.wav")
    syn_style_from_audio(syn_text, model_path_200, audio_f, ref_audio, data_parallel=True)


if __name__ == "__main__":
    test_token_selection()