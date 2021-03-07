import torch
import torch.nn as nn
import torch.nn.functional as F


class ProsodyEncoder(nn.Module):
    def __init__(self, idim, odim, units=256, layers=3):
        super(ProsodyEncoder, self).__init__()
        self.in_ser = torch.nn.ModuleList()
        for layer in range(layers):
            ichans = idim if layer == 0 else units
            ochans = odim if layer == layers - 1 else units
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

    def forward(self, emo_labs):
        for i in range(len(self.in_ser)):
            emo_labs = self.in_ser[i](emo_labs)
        emo_feats = emo_labs
        return emo_feats


class REVLSTMClassifier(nn.Module):
    """docstring for LSTMClassifier"""
    def __init__(self,
                 input_dim,
                 prosody_dim,
                 emo_dim,
                 hidden_dim = 256,
                 n_layers = 2,
                 bidirectional = True,
                 dropout = 0.2,
        ):
        super(REVLSTMClassifier, self).__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.input_dim = input_dim
        self.prosody_dim = prosody_dim
        self.hidden_dim = hidden_dim
        self.emo_dim = emo_dim
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, bias=True,
                           num_layers=self.n_layers, dropout=self.dropout,
                           bidirectional=self.bidirectional)
        self.out = nn.Linear(self.hidden_dim, self.emo_dim)

        self.prosody_enc = ProsodyEncoder(self.emo_dim, self.prosody_dim)

        self.softmax = F.softmax

    def forward(self, input_seq: torch.Tensor = None):
        if len(input_seq.shape) != 3:
            input_seq = input_seq.unsqueeze(0)
        # input_seq =. [1, batch_size, input_size]
        rnn_output, (hidden, _) = self.rnn(input_seq)
        if self.bidirectional:  # sum outputs from the two directions
            rnn_output = rnn_output[:, :, :self.hidden_dim] + \
                         rnn_output[:, :, self.hidden_dim:]
        emo_scores = F.softmax(self.out(rnn_output[0]), dim=1)
        psd_scores = self.prosody_enc(emo_scores)
        return emo_scores, psd_scores

    def inference(self,
                  emo_feats: torch.Tensor=None,
                  emo_labs: torch.Tensor=None):
        # input_seq =. [1, batch_size, input_size]
        if emo_feats is not None:
            rnn_output, (hidden, _) = self.rnn(emo_feats)
            if self.bidirectional:  # sum outputs from the two directions
                rnn_output = rnn_output[:, :, :self.hidden_dim] +\
                            rnn_output[:, :, self.hidden_dim:]
            emo_labs = F.softmax(self.out(rnn_output[0]), dim=1)
        psd_scores = self.prosody_enc(emo_labs)
        return emo_labs, psd_scores