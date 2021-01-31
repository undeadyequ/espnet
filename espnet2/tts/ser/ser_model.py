import sklearn
from joblib import dump, load
import torch


class SER_XGB:
    def __init__(self, clf_path):
        self.clf = load(clf_path)
    def __call__(self, emo_feats):
        audio_prob = self.clf.predict_proba(emo_feats)
        return audio_prob[0]


class REV_SER(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, emo_feats):
        pass

    def inference(self, emo_feats):
        pass


class SER_NN(torch.nn.Module):
    def __init__(self, idim, edim, units=128, layers=2):
        self.ser = torch.nn.ModuleList()

        for layer in range(layers):
            ichans = idim if layer == 0 else units
            ochans = units if layer == 0 else edim
            self.ser += [
                torch.nn.Sequential(
                    torch.nn.Linear(ichans, ochans),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(),
                )
            ]

        for layer in range(layers):
            ichans = edim if layer == 0 else units
            ochans = units if layer == 0 else idim
            self.in_ser += [
                torch.nn.Sequential(
                    torch.nn.Linear(ichans, ochans),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(),
                )
            ]

    def forward(self, emo_feats):

        for i in range(len(self.ser)):
            emo_feats = self.ser[i](emo_feats)

        for i in range(len(self.in_ser)):
            emo_feats = self.in_ser(emo_feats)

        return emo_feats


    def inference(self, emo_feats: torch.Tensor):
        emo_feats = emo_feats.unsqueeze(0)
        return self.forward(emo_feats)


