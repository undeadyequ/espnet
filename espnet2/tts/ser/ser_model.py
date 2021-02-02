import sklearn
from joblib import dump, load
import torch

class SER_XGB(torch.nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

class DNNRevNetwork(torch.nn.Module):
    def __init__(self, idim, edim, units=128, layers=2):
        super(DNNRevNetwork, self).__init__()
        self.ser = torch.nn.ModuleList()

        for layer in range(layers):
            ichans = idim if layer == 0 else units
            ochans = edim if layer == layers - 1 else units
            if layer != layers - 1:
                ser = torch.nn.Sequential(
                    torch.nn.Linear(ichans, ochans),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(),
                )
            else:
                ser = torch.nn.Sequential(
                    torch.nn.Linear(ichans, ochans),
                    torch.nn.Softmax(dim=1)
                )
            self.ser += [ser]

        self.in_ser = torch.nn.ModuleList()
        for layer in range(layers):
            ichans = edim if layer == 0 else units
            ochans = idim if layer == layers - 1 else units
            if layer != layers - 1:
                in_ser = torch.nn.Sequential(
                    torch.nn.Linear(ichans, ochans),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(),
                )
            else:
                in_ser = torch.nn.Sequential(
                    torch.nn.Linear(ichans, ochans),
                    torch.nn.Tanh()
                )
            self.in_ser += [in_ser]

    def forward(self, emo_feats=None, emo_labs=None):
        if emo_feats is not None:
            for i in range(len(self.ser)):
                emo_feats = self.ser[i](emo_feats)
            emo_labs = emo_feats

            for i in range(len(self.in_ser)):
                emo_feats = self.in_ser[i](emo_feats)
        elif emo_labs is not None:
            for i in range(len(self.in_ser)):
                emo_labs = self.in_ser[i](emo_labs)
            emo_feats = emo_labs
        else:
            raise IOError("Input emo_feats or emo_labs")

        return emo_labs, emo_feats

    def inference(self,
                  emo_feats: torch.Tensor=None,
                  emo_labs: torch.Tensor=None):
        if emo_feats is not None:
            emo_feats = emo_feats.unsqueeze(0)
            return self.forward(emo_feats)
        elif emo_labs is not None:
            emo_labs = emo_labs.unsqueeze(0)
            return self.forward(emo_labs=emo_labs)
        else:
            raise IOError("Input emo_feats or emo_labs")