"""
Synthesize audio by

# 1. style_token (Single token or mixed token)
# 2. reference audio

<- synthesis_tacotron_gst

"""
from utils import save_att_png, spectrogram2wav
from scipy.io.wavfile import write
#from Network import Tacotron, Tacotron_gst
from other.Network_git_syn import Tacotron_gst, Tacotron_spec_style
from emoTTS.nets import Tacotron
import torch.nn.functional as F

from Data import get_eval_data, get_eval_data_emo
import sys
import torch
from emoTTS.nets import Hyperparameters as H
from torch.nn import DataParallel


# Synthesis_tacotron
TOKEN_VAR = "module.gst.stl.embed"
GST_VALUE_LINEAR_WEIGHT = "module.gst.stl.attention.W_value.weight"


def synthesis_tacotron(text, model_path, audio_f):
    att_f = audio_f[:-3] + "png"
    print("Start synthesize:{} by tacotron model:{}".format(text, model_path))
    # Load model

    model_tac = Tacotron()
    ckpt = torch.load(model_path)
    model_tac.load_state_dict(ckpt, strict=True)

    # Synthesis
    model_tac.eval()
    text, mel_start = get_eval_data(text)
    mels_hat, mags_hat, att = model_tac(text, mel_start)
    #mels_hat = mels_hat.squeeze().detach().numpy()
    mags_hat = mags_hat.squeeze().detach().numpy()
    att = att.squeeze().detach().numpy()

    # Save result
    save_att_png(att, att_f)
    syn_wav = spectrogram2wav(mags_hat)
    write(audio_f, H.sr, syn_wav)


def get_style_embedding(model_path, token_scale):
    """
    Get style embedding from token_indx or ref_path
    1. Style Transfer
    2. Token Selection
    token_idx:
        0 : use all tokens
        1 : use 1st token
        ...
        10: use 10th token
    :param model_path:
    :return:
    """
    # Load Model

    # model_gst = Tacotron_gst().cuda()   # Use GPU
    model_gst = Tacotron_gst()
    model_gst = DataParallel(model_gst)  # !!! DataParallel append module.***
    # ckpt = torch.load(model_path)
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    # ckpt = torch.load(model_path)
    model_gst.load_state_dict(ckpt, strict=True)
    model_gst.eval()

    model_gst_param = model_gst.state_dict()
    tokens = model_gst_param[TOKEN_VAR]  # keys: (nums_token, E // nums_head)
    v_weights = model_gst_param[GST_VALUE_LINEAR_WEIGHT]  # (E // nums_head, E) ??? Why tanh ???
    out = _compute_style_from_tokens(tokens, v_weights, token_scale)  # (1, 1, E)
    return out


def _compute_style_from_tokens(tokens, v_weights, token_scale):
    """
    compute style from tokens
    :param tokens:
    :param v_weights: Value weights multiply keys
        keys = tanh(tokens)   # ??? Why ???
        Value =  keys * v_weights
    :param token_scale:
    :return:
    """
    keys = F.tanh(tokens).unsqueeze(0).expand(1, -1, -1)  # [1, token_num, E // num_heads]
    value = torch.matmul(keys, torch.t(v_weights))  # (1, token_num, E)
    split_size = H.E // H.nums_head
    value = torch.stack(torch.split(value, split_size, dim=2), dim=0)  # (num_heads, 1, token_num, E // num_heads)
    if token_scale.size() == torch.Size((H.nums_token,)):
        token_scale = torch.t(token_scale)
        token_scale = token_scale.unsqueeze(0).expand(1, -1)
        token_scale = token_scale.unsqueeze(0).expand(1, -1, -1)
        token_scale = token_scale.unsqueeze(0).expand(H.nums_head, -1, -1, -1)  # (num_heads, 1, 1, token_num)
    else:
        print("The token_scale size should be : ", (H.nums_token, 1), " but get :", token_scale.size())
        sys.exit("System exit")
    assert (token_scale.size() == torch.Size((H.nums_head, 1, 1, H.nums_token)))
    out = torch.matmul(token_scale, value)  # (num_heads, 1, 1, E // num_heads)
    out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # (1, 1, E)
    return out


def syn_style_from_audio(text, model_path, audio_f, ref_path=None, data_parallel=True):
    """
    synthesize tacotron style by reference audio

    audio_f      : Output audio file
    data_parallel:
        True : data_parallel used in training, Parameter name : module.encoder.**
        False: data_parallel Not used in training, Parameter name : encoder.**

    """
    att_f = audio_f[:-3] + "png"
    # print("Synthesize TEXT :{} by tacotron_gst model:{}".format(text, model_path))

    # Load Model
    if torch.cuda.is_available():
        model_gst = Tacotron_gst().cuda()   # Use GPU
        ckpt = torch.load(model_path)
    else:
        model_gst = Tacotron_gst()
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    if data_parallel:
        model_gst = DataParallel(model_gst)  # !!! DataParallel append module.***
    model_gst.load_state_dict(ckpt, strict=True)
    model_gst.eval()

    # Load input data
    text, mel_start, ref_mels = get_eval_data_emo(text, ref_path)

    # Synthesis audio
    mels_hat, mags_hat, att = model_gst(text, mel_start, ref_mels)
    mags_hat = mags_hat.squeeze().detach().numpy()
    att = att.squeeze().detach().numpy()
    save_att_png(att, att_f)
    syn_wav = spectrogram2wav(mags_hat)
    write(audio_f, H.sr, syn_wav)


def syn_style_from_token(text, model_path, audio_f, token_scale, data_parallel=True):
    """
    synthesize Tacotron style by Token Selection defined in token_scale
    Load Tacotron_spec_style.

    """
    att_f = audio_f[:-3] + "png"
    # print("Synthesize TEXT :{} by tacotron_gst model:{}".format(text, model_path))

    # Load Model
    if torch.cuda.is_available():
        model_gst = Tacotron_spec_style().cuda()   # Use GPU
        ckpt = torch.load(model_path)
    else:
        model_gst = Tacotron_spec_style()
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    if data_parallel:
        model_gst = DataParallel(model_gst)  # !!! DataParallel append module.***
    model_gst.load_state_dict(ckpt, strict=False)
    model_gst.eval()

    # Load style_emb
    style_emb = get_style_embedding(model_path, token_scale)  # (1, 1, E)
    text, mel_start = get_eval_data(text)

    # Synthesis
    mels_hat, mags_hat, att = model_gst(text, mel_start, style_emb)

    #mels_hat = mels_hat.squeeze().detach().numpy()
    mags_hat = mags_hat.squeeze().detach().numpy()
    att = att.squeeze().detach().numpy()
    save_att_png(att, att_f)
    syn_wav = spectrogram2wav(mags_hat)
    write(audio_f, H.sr, syn_wav)
