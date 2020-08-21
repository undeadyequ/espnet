#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron2 decoder related modules."""

import six
import torch
import torch.nn.functional as F
from espnet.nets.pytorch_backend.rnn.attentions import AttForwardTA

class Prenet(torch.nn.Module):
    """
    Pre-net: 2 layer 256-FN-ReLU-Dropout
    """
    def __init__(self,
                 idim,
                 layers=2,
                 units=256,
                 dropout_rate=0.5):

        super(Prenet, self).__init__()
        self.dropout_rate = dropout_rate
        self.prenet = torch.nn.ModuleList()

        for layer in range(layers):
            input_dim = idim if layer == 0 else units
            self.prenet += [
                torch.nn.Sequential(
                    torch.nn.Linear(input_dim, units),
                    torch.nn.ReLU())
            ]

    def forward(self, x):
        out = x
        for i in range(len(self.prenet)):
            x = F.dropout(self.prenet[i](out), self.dropout_rate)
        return x


class Postnet(torch.nn.Module):
    """
	PostNet: 5_layer conv-5-512 BN-tanh-dropout?(drop not used in paper)  (optional)
    """
    def __init__(self,
                 idim,
                 odim,
                 n_layers=5,
                 n_chans=512,
                 n_filts=5,
                 dropout_rate=0.5,
                 use_batch_norm=True):
        """

        :param idim:
        :param n_layers:
        :param n_chans:
        :param n_filts:
        :param dropout_rate:
        """
        self.postnet = torch.nn.ModuleList()
        for i in range(n_layers):
            ichans = idim if i == 0 else n_chans
            ochans = n_chans if i != n_layers - 1 else odim
            if use_batch_norm:
                self.postnet += [torch.nn.Sequential(
                    torch.nn.Conv1d(ichans,
                                    ochans,
                                    n_filts,
                                    stride=1,
                                    padding=(n_filts - 1) // 2,
                                    bias=False),
                    torch.nn.BatchNorm1d(),
                    torch.nn.Tanh(),
                    torch.nn.Dropout(dropout_rate)
                )]
            else:
                self.postnet += [torch.nn.Sequential(
                    torch.nn.Conv1d(ichans,
                                    ochans,
                                    n_filts,
                                    stride=1,
                                    padding=(n_filts - 1) // 2,
                                    bias=False),
                    torch.nn.Tanh(),
                    torch.nn.Dropout(dropout_rate)
                )]

    def forward(self, x):
        for i in range(len(self.postnet)):
            x = self.postnet[i](x)
        return x


class Decoder(torch.nn.Module):
    """
    Decoder:
        Pre-net: 2 layer 256-FN-ReLU-Dropout
        Lstm: 2_layer uni-Lstm-1024 zoneout-0.1
        Linear Projection: 2_parallel linear
        PostNet: 5_layer conv-5-512 BN-tanh-dropout?(drop not used in paper)  (optional)
    """
    def __init__(
            self,
            idim,
            odim,
            att,
            dlayers,
            dunits,
            prenet_layers,
            prenet_units,
            postnet_layers,
            postnet_chans,
            postnet_filters,
            use_batch_norm=True,
            use_concate=True,
            cumulate_att_w=True,
            output_activation_fn=None,
            dropout_rate=0.5,
            zoneout_rate=0.1,
            reduction_factor=1
    ):
        """
        :param idim: Dimension of input
        :param odim:
        :param att:
        :param layers:
        :param dunits:
        :param dropout_rate:
        """
        self.idim = idim
        self.odim = odim
        self.dlayers = dlayers
        self.reduction_factor = reduction_factor
        self.prenet = Prenet(idim, prenet_layers, prenet_units, dropout_rate)
        self.att = att
        self.output_activation_fn = output_activation_fn
        self.cumulate_att_w = cumulate_att_w

        if isinstance(att, AttForwardTA):
            self.use_att_extra_inputs = True
        else:
            self.use_att_extra_inputs = False

        if postnet_layers > 0:
            self.postnet = Postnet(
                idim=odim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filters,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm)

        self.lstm = torch.nn.ModuleList()

        for layer in range(dlayers):
            iunits = idim + prenet_units if layer == 0 else dunits
            lstm  = torch.nn.LSTMCell(iunits, dunits)
            if zoneout_rate > 0.0:
                pass
            self.lstm += [torch.nn.LSTMCell(iunits, dunits)]

        iunits = idim + dunits if use_concate else dunits
        self.feat_out = torch.nn.Linear(iunits, odim * reduction_factor, bias=False) # why bias = False???
        self.prob_out = torch.nn.Linear(iunits, reduction_factor)

    def _zero_state(self, hs, units):
        return torch.zeros((hs.size(0), units))

    def calculate_all_attentions(self, hs, hlens, ys):
        att_ws = 0
        return att_ws

    def forward(self,
                hs,
                hlens,
                ys):
        """

        :param hs:    (B, Tmax, idim)
        :param hlens:
        :param ys:    (B, Lmax, odim)
        :return:
        """
        # target modification
        if self.reduction_factor > 1:
            seq_cut = int(ys.size(1) / self.reduction_factor) * self.reduction_factor
            ys_cut = ys[:, seq_cut, :]
            ys = ys[:, :ys_cut, :]
            ys_reduction = []
            for i in range(self.reduction_factor):
                ys_reduction += ys[:, i::self.reduction_factor]
            ys = torch.cat(ys_reduction, dim=2)

        # initial attention
        pre_att_w = None

        # initial hidden status of decoder
        c_list, z_list = [], []
        for _ in range(self.dlayers):
            c_list += [self._zero_state(hs, self.lstm[0].hidden_state)]
            z_list += [self._zero_state(hs, self.lstm[0].hidden_state)]
        prev_out = hs.new_zeros(hs.size(0), self.odim)

        # loop for output sequence
        outs, logits, att_ws = [], [], []
        for y in ys.transpose(0, 1):
            # attention
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(hs, hlens, z_list[0], pre_att_w, prev_out)  # dim(z_list[0]) == (B, hidden_size), hidden_state == dim(hs) ???
            else:
                att_c, att_w = self.att(hs, hlens, z_list[0], pre_att_w)

            # lstm
            prenet_out = self.prenet(prev_out)
            xs = torch.cat([att_c, prenet_out], dim=1)
            for i in range(len(self.lstm)):
                if i == 0:
                    z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
                else:
                    z_list[i], c_list[i] = self.lstm[i](z_list[i-1], (z_list[i-1], z_list[i-1]))

            # 2 Linear projection
            zcs = (torch.cat([z_list[-1], att_c], dim=1)
                   if self.use_concate
                   else z_list[-1])
            outs += [self.feat_out(zcs)]   # (B, odim) * Lmax
            logits += [self.prob_out(zcs)]
            att_ws += [att_w]

            if self.cumulate_att_w and pre_att_w is not None:
                pre_att_w += att_w
            else:
                pre_att_w = att_w
            prev_out = y

        logits = torch.cat(logits, dim=1)  # (B, Lmax//red_f)
        before_outs = torch.cat(outs, dim=2) # (B, odim*red_f, Lmax//red_f)
        att_ws = torch.cat(att_ws, dim=1) # (B, Lmax//red_f, Tmax)

        if self.reduction_factor > 1:
            # logits and att_ws !!!
            before_outs = before_outs.view(
                before_outs.size(0), self.odim, -1
            )  # (B, odim, Lmax)

        if self.postnet is not None:
            after_outs = before_outs + self.postnet(before_outs)  # (B, odim, Lmax)
        else:
            after_outs = before_outs

        if self.output_activation_fn:
            before_outs = self.output_activation_fn(before_outs)
            after_outs = self.output_activation_fn(after_outs)

        return before_outs, after_outs, logits, att_ws


    def inference(
            self,
            h,
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=False,
            backward_window=None,
            forward_window=None
    ):
        # setup

        maxlen = int(h.size[0] * maxlenratio)
        minlen = int(h.size[0] * minlenratio)

        #

        outs, att_ws, probs = [], [], []
        idx = 0
        while True:
            # updated index
            idx += self.reduction_factor

            # decoder calculation
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(hs, ilens, z_list[0], prev_att_w, prev_out,
                                        last_attended_idx=last_attended_idx,
                                        backward_window=backward_window,
                                        forward_window=forward_window)
            else:
                att_c, att_w = self.att(hs, ilens, z_list[0], prev_att_w,
                                        last_attended_idx=last_attended_idx,
                                        backward_window=backward_window,
                                        forward_window=forward_window)
            out = self.linear(att_c)
            prob = self.sigmoid(self.linear(out))

            outs += [out]
            probs += [prob]
            att_ws += [att_w]

            # check whether to finish generation
            if int(sum(probs[-1] >= threshold)) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                outs = torch.cat(outs, dim=2)  # (1, odim, L)
                if self.postnet is not None:
                    outs = outs + self.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                probs = torch.cat(probs, dim=0)
                att_ws = torch.cat(att_ws, dim=0)
                break


        if self.output_activation_fn is not None:
            outs = self.output_activation_fn(outs)

        return outs, probs, att_ws
