"""Attention modules for RNN."""

import math
import six

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device

class AttLocRec(torch.nn.Module):
    def __init__(self,
                 eprojs, dunits, att_dim, aconv_chans, aconv_filt):
        self.W = torch.nn.Linear(dunits, att_dim)
        self.V = torch.nn.Linear(eprojs, att_dim)
        self.U = torch.nn.Linear(dunits, att_dim)
        self.att_dim = att_dim

        self.att_conv2d = torch.nn.Conv2d(
            1,
            aconv_chans,
            (1, 2 * aconv_filt + 1),
            padding=(0, aconv_filt),
            bias=False)
        self.att_rnn = torch.nn.LSTMCell(aconv_chans, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.pre_compute_enc_h = None
        self.h_length = None
        self.enc_hs = None
        self.mask = None

    def reset(self):
        self.pre_compute_enc_h = None
        self.h_length = None
        self.enc_hs = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_states, scaling=2.0):
        """
        :param enc_hs_pad:    (B, T_max, D_enc)
        :param enc_hs_len:(B, )
        :param dec_z:     (B, D_dec)
        :param att_prev_states: ((B, T_max), (B, att_dim), (B, att_dim)))
        :param scaling:
        :return:
            att_c:  (B, D_enc)
            att_w and att_states:  ((B, T_max), (B, att_dim), (B, att_dim)) # (w, (hx, cx))
        """
        b = len(enc_hs_pad)
        if self.pre_compute_enc_h is None:
            self.enc_hs = enc_hs_pad
            self.h_length = self.enc_hs.size(1)
            self.pre_compute_enc_h = self.V(enc_hs_pad)

        if att_prev_states is None:
            att_prev = None
            att_h = enc_hs_pad.new_zeros(b, self.att_dim)
            att_c = enc_hs_pad.new_zeros(b, self.att_dim)
            att_states = (att_h, att_c)
        else:
            att_prev, att_states = att_prev_states[0], att_prev_states[1]

        ## Attention RNN
        # Input: B * 1 * 1 * T -> B * C * 1 * T -> B * C
        att_conv = F.relu(self.att_conv2d(
                att_prev.view(b, 1, 1, self.h_length)
        ))
        att_conv = F.max_pool2d(att_conv, att_conv.size(2))
        att_h, att_c = self.att_rnn(att_conv, att_states)    #（B, att_dim）
        dec_z_tiled = self.U(dec_z).view(b, 1, self.att_dim)

        ## Attention Score
        score = self.gvec(F.tanh(
            self.W(att_h).unsqueeze(1) +
            self.pre_compute_enc_h +
            dec_z_tiled
        )).squeeze(2)  # (B, T_max)    ??? self.W is not used
        if self.mask:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        score.masked_fill_(self.mask, -float("inf"))

        ## Attention Weight (B, T_max)
        w = F.softmax(scaling * score, dim=1)

        ## Context vector
        c_v = torch.sum(enc_hs_pad * w.view(b, self.h_length, 1), dim=1)
        return c_v, (w, (att_h, att_c))

