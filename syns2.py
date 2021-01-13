import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech
from librosa.feature import melspectrogram, mfcc
import librosa
import os


## syn txt file
# provided pre-trained model
# own model


d = ModelDownloader()
print(d.download_and_unpack("kan-bayashi/vctk_gst_tacotron2"))
text2speech = Text2Speech(**d.download_and_unpack("kan-bayashi/vctk_gst_tacotron2"))

#ref_audio = "/home/rosen/Project/espnet/egs/blizzard13/tts2_gst/ref_audio/CB-JE-14-37.wav"
#ref_audio = "ref_audio/speaker/p245_001_m1.wav"
ref_audio = "ref_audio/emotion/CB-SCA-01-100_normal.wav"


y, sr = librosa.load(ref_audio)
print(y.shape)
#mels = melspectrogram(y, sr, n_mels=80).T

#print("mel_shape:", mels.shape)

text = "I hate work at home."
speech, *_ = text2speech(text, y)

import string
exclude = set(string.punctuation)

text_space = text.replace(" ", "_").translate(str.maketrans('_', '_', string.punctuation))


out = "result_vctk_gst_tacotron2/emotion/" + text_space + os.path.basename(ref_audio)

soundfile.write(out, speech.numpy(), text2speech.fs, "PCM_16")