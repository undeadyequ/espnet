from utils.experiment.syns_audio import syn_audio
import soundfile
from espnet2.bin.tts_inference import Text2Speech
from librosa.feature import melspectrogram, mfcc
import librosa
import os
import string
import argparse
import torch
# provided pre-trained model
## saved in ~/Project/espnet/tools/venv/envs/espnet/lib/python3.8/site-packages/espnet_model_zoo/d51358bee8acf0087b09afface2eb09
from utils.extract_emo_features import extract_emo_feature
# own model
import matplotlib.pyplot as plt


emo_contrl_dict = {
    "sig_m": 0,
    "sig_s": 1

# sig_m, sig_std, eng_m, eng_std, sil, harm, pitch_mean, pitch_std
}

class ExpStyleTTS:
    """
    Experiment on stylish TTS model by 3 tests.

    result folder
    - eval_dataset_modelName
        - ref_emo
            - ang
            - hap
        - emo_contrl
            - ang
            - hap
        - psd_contrl
            - pitch
            - energy
            - ...
    """
    def __init__(self, model, config, text, out_dir, vocoder_conf: dict = None):
        self.model = model
        self.config = config
        self.text = text
        self.out_dir = out_dir

        self.default_contrl_value = []

        if not os.path.isdir(self.out_dir):
            os.system("mkdir {}".format(out_dir))

        if vocoder_conf is None:
            vocoder_conf = {"n_fft": 1024, "n_shift": 256, "fs": 22050, "n_mels": 80}
        self.text2speech = Text2Speech(train_config=config, model_file=model, vocoder_conf=vocoder_conf)
        self.text_nospacepunc = self.text.replace(" ", "_").translate(str.maketrans('_', '_', string.punctuation))

        self.ref_dir = "/home/rosen/Project/espnet/ref_audio/emotion"
        self.emo_labs = ["ang", "hap"]
        self.psd_contrl = {"pitch": [-1, 0, 1], "eng": [-1, 0, 1]}

    def exp_ref_audio(self, ref_dir):
        if ref_dir is not None:
            self.ref_dir = ref_dir
        print("ref_dir", self.ref_dir)
        for ref in os.listdir(self.ref_dir):
            y, sr = librosa.load(os.path.join(self.ref_dir, ref))
            speech, *_ = self.text2speech(self.text, speech=y)
            out = os.path.join(self.out_dir, self.text_nospacepunc.lower() + os.path.basename(ref))
            soundfile.write(out, speech.numpy(), self.text2speech.fs, "PCM_16")

    def exp_emo_lab(self, emo_labs):
        if type(emo_labs) == type(""):
            emo_labs = [float(l) for l in emo_labs.split(" ")]

        speech, *_ = self.text2speech(self.text, emo_labs=self.emo_labs)
        extract_emo_feature(speech,
                            normlaize=True,
                            min_max_stats_f="/home/Data/blizzard2013_part_preprocess/dump/emo_feats/feats_stats.csv")

        out = os.path.join(self.out_dir, self.text_nospacepunc, torch.Tensor(emo_labs))
        soundfile.write(out, speech.numpy(), self.text2speech.fs, "PCM_16")


    def exp_psd_contrl(self, psds):
        """

        Args:
            psds: [psd_n, 8]

        Returns:


        """
        # Find exp_emof
        exp_emof = 0
        if len(psds) > 1:
            diff_l = list(set(psds[0]) - set(psds[1]))
            exp_emof = psds[0].index(diff_l[0])
        inp_exp_emofs = []
        syn_exp_emofs = []

        # show input emo_contrls and synthesized emo_contrls
        for input_emof in psds:
            print("input emo_contrl:{}".format(input_emof))
            speech, *_ = self.text2speech(self.text, emo_feats=torch.Tensor(input_emof))
            # extract feature
            syn_emof = extract_emo_feature(speech.numpy(), normlaize=True)[0]
            print("syn emo_contrl:{}".format(syn_emof))

            inp_exp_emofs.append(input_emof[exp_emof])
            syn_exp_emofs.append(syn_emof[exp_emof])
            # Syn wav
            out = os.path.join(self.out_dir, self.text_nospacepunc.lower() + "_" + str(exp_emof) + "_"
                               + str(input_emof[exp_emof]) + ".wav")
            soundfile.write(out, speech.numpy(), self.text2speech.fs, "PCM_16")

        # draw picture only for comparing emo_contrl
        draw_line(inp_exp_emofs, syn_exp_emofs, exp_emof)




def draw_line(inp_exp_emofs, syn_exp_emofs, exp_emof):
    """
    Draw line from points
    Args:
        pts:

    Returns:

    """
    plt.plot(inp_exp_emofs, syn_exp_emofs)
    plt.xlabel("input_emof_{}".format(exp_emof))
    plt.ylabel("syn_emof_{}".format(exp_emof))
    out = "exp_line{}.png".format(exp_emof)
    plt.savefig(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--text", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--ref_dir", type=str, default=None)
    parser.add_argument("--emo_feats", nargs='+', default=None)
    parser.add_argument("--emo_labs", type=str, default=None)

    args = parser.parse_args()
    print("outd_dir", args.out_dir)
    print("text", args.text)
    exp = ExpStyleTTS(args.model, args.config, args.text, args.out_dir)

    #exp.exp_ref_audio(args.ref_dir)
    if args.emo_feats is not None:
        emo_feats=[]
        for psd in args.emo_feats:
            psds_l = [float(l) for l in psd.split(" ")]
            emo_feats.append(psds_l)
        exp.exp_psd_contrl(emo_feats)
    #exp.exp_emo_lab(args.emo_labs)

