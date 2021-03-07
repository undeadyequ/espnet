from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from utils.extract_emofeats import extract_emofeats_from_scp
from utils.extract_txtfeats import extract_txtfeats_from_txt
import pickle
import numpy as np
import argparse
import pandas as pd
import h5py


def extract_combinedfeats_from_scp(id_wav_f, id_txt_f, corpus_f=None, out_pkl="out.pkl"):
    """
    Extract combined feats into pkl file
    Args:
        input_csv:
        input_scp:
        corp_csv:
        out_pkl:
        vis:

    Returns:

    """
    temp_audiofts = "temp_emofts.csv"
    temp_txtfts = "temp_txtfts.pkl"

    # Extract
    extract_emofeats_from_scp(id_wav_f, temp_audiofts)
    extract_txtfeats_from_txt(id_txt_f, corp_csv=corpus_f, out_pkl=temp_txtfts)


def combined_audio_txt_file(audio_fts_f, txtfts_pkl, out):
    """
    id audo_fts txt_fts.
    Args:
        audio_fts_f:
        txtfts_pkl:
        out_pkl:

    Returns:
        np: (n_sample, id_dim[1] + tfidf_dim + audiofts_dim[8] )   => (, 2302)
    """

    # Combine
    audfits = pd.read_csv(audio_fts_f, header=None, sep=" ")
    with open(txtfts_pkl, "rb") as f:
        txtfts = pickle.load(f)

    combfits = np.concatenate((np.array(audfits[audfits.columns[1:]], dtype=np.float32), txtfts), axis=1)
    ids = np.array(audfits[audfits.columns[0]])
    # Save
    if out[-3:] == "pkl":
        with open(out, "wb") as f:
            pickle.dump(combfits, f)
    elif out[-4:] == "hdf5":
        with h5py.File(out, "w") as f:
            for i, r in enumerate(combfits):
                f.create_dataset(ids[i], data=r)
    else:
        print("combined fts must save in pkl and hdf5, but got {}".format(out))


def combine_id_emo_fts(id_fts_f, id_emo_f, id_emo_fts_f=""):
    id_emo = pd.read_csv(id_emo_f, header=None, sep=" ")
    id_fts = pd.read_csv(id_fts_f, header=None, sep=" ")
    emotion_dict = {'ang': 0,
                    'hap': 1,
                    'exc': 2,
                    'sad': 3,
                    'fru': 4,
                    'fea': 5,
                    'sur': 6,
                    'neu': 7,
                    'xxx': 8,
                    'oth': 8}
    id_emo = id_emo.iloc[:, 0:2]
    id_emo.iloc[:, 1] = id_emo.iloc[:, 1].replace(emotion_dict)

    id_emo_fts = id_emo.merge(id_fts, on=0)
    id_emo_fts.to_csv(id_emo_fts_f, index=False, sep=",", header=None)


def modify_id_emo(id_emo_f, id_emo_mod):
    """
    id emo contr ....
    Args:
        id_emo_f:
        id_emo_mod:

    Returns:

    """
    id_emo = pd.read_csv(id_emo_f, header=None, sep=" ")
    id_emo = id_emo.iloc[:, 0:2]

    emotion_dict = {'ang': 0,
                    'hap': 1,
                    'exc': 2,
                    'sad': 3,
                    'fru': 4,
                    'fea': 5,
                    'sur': 6,
                    'neu': 7,
                    'xxx': 8,
                    'oth': 8}
    id_emo.replace({"1": emotion_dict})

    id_emo.to_csv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str)
    parser.add_argument("--sr", default=22050)
    combine_id_emo_fts()


