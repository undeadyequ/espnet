#!/usr/bin/env python3

#from utils.dsp import silenceremove
from utils.dsp.extract_emo_features import extract_emo_feature_modify
import pytest
import os
import pandas as pd

from utils.extract_emofeats import extract_emofeats_from_scp, normalize_audio_fts
from utils.extract_txtfeats import extract_txtfeats_from_txt
from utils.extract_combinedfeats import extract_combinedfeats_from_scp, combined_audio_txt_file
from utils.extract_combinedfeats import *
from utils.dsp.prepare_data_modified import prepare_text_data



emo_num = { 'ang': 0, 'hap': 1, 'sad': 2, 'fea': 3, 'sur': 4, 'neu': 5 }


@pytest.mark.parametrize("input_wav", ["test_wav/Ses01F_impro01_F000_no_interval_short.wav",
                                       "test_wav/Ses01F_impro04_F002_interval_long.wav"])
def test_silenceremove(input_wav):
    out_wav = input_wav[:-4] + "_remove_silence.wav"
    os.system("python ../utils/dsp/silenceremove.py {} {} {}".format(1, input_wav, out_wav))


def test_audio_emo_feats():
    """

    Returns:

    """
    dir = "test_wav/emotion"
    input_wav = ["hap/Ses01F_impro03_F004.wav",
                 "hap/Ses01F_impro03_F001.wav",
                 "hap/Ses01F_impro03_F000.wav",
                 "ang/Ses01F_impro05_F018.wav",
                 "ang/Ses01F_impro05_F017.wav",
                 "ang/Ses01F_impro05_F016.wav",
                 "sad/Ses01M_impro06_F005.wav",
                 "sad/Ses01M_impro06_F004.wav",
                 "sad/Ses01M_impro06_F006.wav"]
    input_wav = [dir + "/" + p for p in input_wav]
    outs = []


    for a in input_wav:
        out = []
        id = os.path.basename(a)[:-4]
        emo = os.path.split(os.path.split(a)[0])[1]
        emofts = extract_emo_feature_modify(a, sr=16000)
        out.append(id)
        out.append(int(emo_num[emo]))
        out.extend(emofts)
        outs.append(out)

    assert len(outs[0]) == 3 + 7
    colnames = ["id", "emo",
                "rmse", "rmse_std", "rmse_range",
                "harmonic", "harmonic_std",
                "pitch", "pitch_std", "pitch_range"]

    outs_pd = pd.DataFrame(outs, columns=colnames)

    temp_f = "test_csv/temp_wav_emofts.csv"
    outs_pd.to_csv(temp_f, index=False)


#@pytest.mark.parametrize("input_csv, out_csv", ["test_csv/wav.csv", "test_csv/wav_out.csv"])
def test_extract_emofeats_from_scp(input_csv, out_csv):
    extract_emofeats_from_scp(input_csv, out_csv)

@pytest.mark.parametrize("input_csv, out_pkl", ["test_csv/text.csv", "test_csv/text_out.pkl"])
def test_extract_txtfeats_from_txt(input_csv, out_pkl):
    extract_txtfeats_from_txt(input_csv, corp_csv=None, out_pkl=out_pkl)

@pytest.mark.parametrize("audio_fts, txt_fts, out_pkl", ["test_csv/index/wav_out.csv", "test_csv/index/text_out.pkl",
                                                         "test_csv/index/out.pkl"])
def test_extract_combinefeats_from_scp(audio_fts, id_emo_file, txt_fts, out_pkl):
    combined_audio_txt_file(audio_fts, id_emo_file, txt_fts, out_pkl)

def test_basic():
    input_csv_1 = "./test_csv/iemocap/wav.csv"
    out_csv_1 = "./test_csv/iemocap/wav_out.csv"

    input_csv_2 = "./test_csv/blizzard13/id_pth.csv"
    out_csv_2 = "./test_csv/blizzard13/id_fts_out.csv"

    input_csv_3 = "./test_csv/blizzard13/id_pth_dev.csv"
    out_csv_3 = "./test_csv/blizzard13/id_fts_dev_out.csv"

    extract_emofeats_from_scp(input_csv_1, out_csv_1)
    extract_emofeats_from_scp(input_csv_2, out_csv_2)
    extract_emofeats_from_scp(input_csv_3, out_csv_3)


def test_basic_2():
    input_csv_3 = "./test_csv/blizzard13/id_pth_dev.csv"
    out_csv_3 = "./test_csv/blizzard13/id_fts_dev_out.csv"

    extract_emofeats_from_scp(input_csv_3, out_csv_3)



def test_iemocap_extraction():
    """
    Only for iemocap
    Returns:

    """
    wav_scp = "./test_csv/iemocap/wav.csv"
    wav_fts = "./test_csv/iemocap/id_fts.csv"
    wav_norm_fts = "./test_csv/iemocap/wav_norm_out.csv"
    normal_stats = "./test_csv/iemocap/normal_stats.csv"
    id_emo_file = "./test_csv/iemocap/emo.csv"

    txt_scp = "test_csv/iemocap/text.csv"
    cor_scp = "test_csv/iemocap/text_corp.csv"
    txt_out = "test_csv/iemocap/text_out.pkl"

    comb_out = "test_csv/iemocap/comb_out.hdf5"
    id_emo_fts = "test_csv/iemocap/id_emo_fts.csv"

    #test_extract_emofeats_from_scp(wav_scp, wav_fts)
    #normalize_audio_fts(wav_fts, method="quantile", out_csv=wav_norm_fts, out_stats_csv=normal_stats)

    #extract_txtfeats_from_txt(txt_scp, corp_csv=cor_scp, out_pkl=txt_out)
    combined_audio_txt_file(wav_norm_fts, txt_out, comb_out)

    #audio2text_f = "/home/rosen/Project/espnet/test_utils/test_csv/audiocode2text.pkl"
    #with open(audio2text_f, "rb") as f:
    #    audio2text = pickle.load(f)
    #prepare_text_data(audio2text, id_emo_fts)


def create_combine_id_fts_tfidf(id_wav, id_txt, txt_corp):
    """
    Extract from id_wav and id_txt, like blizzard13
    Args:
        id_wav:
        fts_norm_stats:
        id_txt:
        txt_corp:

    Returns:

    """
    id_wav_base = id_wav[:-4]
    wav_fts = id_wav_base + "_fts.csv"
    wav_fts_norm = id_wav_base + "_fts_norm.csv"
    out_fts_norm_stats = id_wav_base + "_fts_norm_stats.csv"

    id_txt_base = id_txt[:-4]
    id_txt_pkl = id_txt_base + "_tfidf.pkl"
    comb_hdf5 = id_wav_base + "_fts_norm_tfidf.hdf5"

    #extract_emofeats_from_scp(id_wav, wav_fts)
    normalize_audio_fts(wav_fts, method="quantile", out_csv=wav_fts_norm, out_stats_csv=out_fts_norm_stats)
    extract_txtfeats_from_txt(id_txt, corp_csv=txt_corp, out_pkl=id_txt_pkl)
    combined_audio_txt_file(wav_fts_norm, id_txt_pkl, comb_hdf5)


def test_blizzard13_extraction():
    id_wav = "./test_csv/blizzard13/id_pth_dev.csv"
    id_fts = "./test_csv/blizzard13/id_fts_dev.csv"
    id_fts_norm = "./test_csv/blizzard13/id_fts_norm.csv"
    normal_stats = "./test_csv/iemocap/normal_stats.csv"

    id_txts = "./test_csv/blizzard13/text.csv"
    cor_scp = "test_csv/iemocap/text_corp.csv"   #
    txt_out = "test_csv/blizzard13/text_out.pkl"

    comb_out = "test_csv/blizzard13/comb_out.hdf5"

    extract_emofeats_from_scp(id_wav, id_fts)
    normalize_audio_fts(id_fts, method="quantile", out_csv=id_fts_norm)

    extract_txtfeats_from_txt(id_txts, corp_csv=cor_scp, out_pkl=txt_out)
    combined_audio_txt_file(id_fts_norm, txt_out, comb_out)


if __name__ == '__main__':
    #test_iemocap_extraction()
    #test_blizzard13_extraction()
    #test_audio_emo_feats()
    #test_blizzard13_extraction()
    #test_basic()
    #test_basic_2()

    # dev
    id_wav = "/home/rosen/Project/espnet/test_utils/test_csv/blizzard13/id_pth_dev.csv"
    id_txt = "/home/rosen/Project/espnet/test_utils/test_csv/blizzard13/text_dev.csv"
    txt_corp = "/home/rosen/Project/espnet/test_utils/test_csv/blizzard13/text_corp.csv"
    create_combine_id_fts_tfidf(id_wav, id_txt, txt_corp)

