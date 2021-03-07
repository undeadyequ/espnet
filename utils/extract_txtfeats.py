import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import pickle


def extract_txtfeats_from_txt(input_csv, corp_csv=None, out_pkl=None, method="tfidf"):
    """
    Extract tfidf feats by corp_csv
    Args:
        input_csv: header (id transcription)
        corp_csv:  if None, use input_csv self as corp_csv
        out_pkl:
        method:

    Returns:

    """
    if corp_csv is None:
        corp_csv = input_csv

    sents_in = []
    with open(input_csv, "r") as f:             #  pd.read_csv not works since space separater
        for row in f:
            id, sent = row.split(" ", 1)
            sents_in.append(sent.rstrip())

    sents_cor = []
    with open(corp_csv, "r") as f:
        for row in f:
            id, sent = row.split(",", 1)
            sents_cor.append(sent.rstrip())

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

    corp_fts = tfidf.fit_transform(sents_cor).toarray()
    inpt_fts = tfidf.transform(sents_in).toarray()   # ck if inpt_fts == corp_fts if corp=None

    if out_pkl is not None:
        with open(out_pkl, "wb") as f:
          pickle.dump(inpt_fts, f)
    return inpt_fts


## old
"""
def extract_txtfeats_from_scp(
        txt_scp: str,
        out_csv: str
):
    pd_path = pd.read_csv(txt_scp, delim_whitespace=True, header=None)

    txt_feats_pd = pd.DataFrame()
    txt_feats = []

    for _, row in pd_path.iterrows():
        audio_id = row[0]
        audio_trscpt = row[1]
        txtfeats = extract_txt_feature(audio_trscpt)

        txtfeats_row = [audio_id, txtfeats]
        txt_feats.append(txtfeats_row)
    txt_feats_pd = txt_feats_pd.append(txt_feats)
    txt_feats_pd.to_csv(out_csv, sep=" ", header=False, index=False)

"""





"""
def extract_txt_feature(txt, method=""):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words='english')

    features = tfidf.fit_transform(df.transcription).toarray()

    labels = df.label
    print(features.shape)
"""


if __name__ == '__main__':
    pass

