import sklearn
from joblib import dump, load

class SER_XGB:
    def __init__(self, clf_path):
        self.clf = load(clf_path)
    def __call__(self, emo_feats):
        audio_prob = self.clf.predict_proba(emo_feats)
        return audio_prob[0]