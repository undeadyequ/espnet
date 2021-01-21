from joblib import dump, load

import numpy as np

from joblib import dump, load

emotion_dict = {'ang': 0,
                'hap': 1,
                'sad': 2,
                'fea': 3,
                'sur': 4,
                'neu': 5}

def cls_emo_pretrained_ml(
        clf: str,
        emo_feats):
    audio_prob = clf.predict_proba(emo_feats)[0]
    emo_res, emo_prob = _explain_audio_prob(audio_prob, emotion_dict)
    return emo_res, audio_prob


def _explain_audio_prob(audio_prob, emo_dict):
    """
    audio_prob: np [n_features,]
    """

    audio_prob_max = 0
    emo_res = []
    if len(audio_prob) != len(emo_dict.keys()):
        print("the emo_dict should be equal with audio_prob")
    else:
        audio_prob_max = np.max(audio_prob)
        emo_res = list(emo_dict.keys())[list(audio_prob).index(audio_prob_max)]
    return emo_res, audio_prob_max