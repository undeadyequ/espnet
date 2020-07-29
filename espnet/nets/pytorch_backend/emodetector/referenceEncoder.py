import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class ReferenceEncoder(nn.Module):
    '''
    idim :
    inputs --- [N, Ty, n_mels]  mels
    outputs --- [N, ref_enc_gru_size]
    '''
    def __init__(self,
                 idim,
                 ref_emb_dim=512,
                 ref_enc_filters=(32, 32, 64, 64, 128, 128),
                 use_batch_norm=True,
                 drop_rate=0.5):
        """ Initialize ReferenceEncoder encoder module.
        Conv2d + blstm      //luo0 good?
        Args:
            idim     : n_mels
            ref_emb_dim:
            ref_enc_filters ()
        """
        self.ref_enc_filters = list(ref_enc_filters)
        self.n_mels = idim
        self.eunits = ref_emb_dim

        super(ReferenceEncoder, self).__init__()
        K = len(self.ref_enc_filters)
        filters = [1] + self.ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]        # luo1 What about kernel-like parameter
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=self.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(self.n_mels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=self.ref_enc_filters[-1] * out_channels,
                          hidden_size=self.eunits,
                          batch_first=True)

    def forward(self, inputs):
        N = inputs.size(0)
        #print("reference input size:", inputs.size)
        out = inputs.view(N, 1, -1, self.n_mels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)

    def inference(self, inputs):
        return self.forward(inputs)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L