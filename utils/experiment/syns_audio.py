import soundfile
from espnet2.bin.tts_inference import Text2Speech
from librosa.feature import melspectrogram, mfcc
import librosa
import os
import string
# provided pre-trained model
## saved in ~/Project/espnet/tools/venv/envs/espnet/lib/python3.8/site-packages/espnet_model_zoo/d51358bee8acf0087b09afface2eb09

# own model


def syn_audio(model, config, text, ref_audio, out_dir):
    text2speech = Text2Speech(train_config=config, model_file=model)
    y, sr = librosa.load(ref_audio)
    # mels = melspectrogram(y, sr, n_mels=80).T
    # print("mel_shape:", mels.shape)

    speech, *_ = text2speech(text, y)
    exclude = set(string.punctuation)

    text_nospace = text.replace(" ", "_").translate(str.maketrans('_', '_', string.punctuation))
    out = os.path.join(out_dir, text_nospace, os.path.basename(ref_audio))
    soundfile.write(out, speech.numpy(), text2speech.fs, "PCM_16")


if __name__ == '__main__':
    model = "/home/rosen/Project/espnet/egs2/blizzard2013/tts1/blizzard2013_part_preprocess/exp/tts_train_fbank_phn_tacotron_g2p_en_no_space/120epoch.pth"
    config = "/home/rosen/Project/espnet/egs2/blizzard2013/tts1/conf/train.yaml"
    # ref_audio = "/home/rosen/Project/espnet/egs/blizzard13/tts2_gst/ref_audio/CB-JE-14-37.wav"
    # ref_audio = "ref_audio/speaker/p245_001_m1.wav"
    ref_audio = "ref_audio/emotion/CB-SCA-01-100_normal.wav"
    text = "I hate work at home."
    out_dir = "result_blizzard13_gst_tacotron2/emotion/"

    syn_audio(model, config, text, ref_audio, out_dir)
