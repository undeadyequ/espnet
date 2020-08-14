#!/usr/bin/env python3

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
from __future__ import division

import numpy as np
import pytest
import torch
import sys
from argparse import Namespace

from espnet.nets.pytorch_backend.e2e_tts_tacotron_gst import Tacotron2_GST
from espnet.nets.pytorch_backend.nets_utils import pad_list


def make_tac2_gst_args(**kwargs):
    defaults = dict(
        # gst
        style_embed_dim=32,
        ref_embed_dim=32,
        # Tacotron2
        use_speaker_embedding=False,
        spk_embed_dim=None,
        embed_dim=32,
        elayers=1,
        eunits=32,
        econv_layers=2,
        econv_filts=5,
        econv_chans=32,
        dlayers=2,
        dunits=32,
        prenet_layers=2,
        prenet_units=32,
        postnet_layers=2,
        postnet_filts=5,
        postnet_chans=32,
        output_activation=None,
        atype="location",
        adim=32,
        aconv_chans=16,
        aconv_filts=5,
        cumulate_att_w=True,
        use_batch_norm=True,
        use_concate=True,
        use_residual=False,
        dropout_rate=0.5,
        zoneout_rate=0.1,
        reduction_factor=1,
        threshold=0.5,
        maxlenratio=5.0,
        minlenratio=0.0,
        use_cbhg=False,
        spc_dim=None,
        cbhg_conv_bank_layers=4,
        cbhg_conv_bank_chans=32,
        cbhg_conv_proj_filts=3,
        cbhg_conv_proj_chans=32,
        cbhg_highway_layers=4,
        cbhg_highway_units=32,
        cbhg_gru_units=32,
        use_masking=True,
        use_weighted_masking=False,
        bce_pos_weight=1.0,
        use_guided_attn_loss=False,
        guided_attn_loss_sigma=0.4,
        guided_attn_loss_lambda=1.0,
    )
    defaults.update(kwargs)
    return defaults


def make_tac2_gst_infer_args(**kwargs):
    defaults = dict(
        threshold=0.5,
        maxlenratio=5.0,
        minlenratio=0.0,
        use_att_constraint=False,
        backward_window=1,
        forward_window=3,
    )
    defaults.update(kwargs)
    return defaults


def prepare_inputs(bs, idim, odim, maxin_len, maxout_len,
                   spk_embed_dim=None, spc_dim=None, device=torch.device("cpu")):
    """
    Prepare inputs like below
    batch = {
    "xs": xs,       (B, Tmax)
    "ilens": ilens, (B, )
    "ys": ys,       (B, Lmax, odim)
    "rs": ys,
    "olens": olens, (B, )
    "labels": labels(B, Lmax)
    "spembs": spembs(B, spk_embed_dim)
    "extras": spctrm(B, Lmax, spc_dim)
    }
    """
    ilens = np.sort(np.random.randint(1, maxin_len, bs))[::-1].tolist()
    olens = np.sort(np.random.randint(3, maxout_len, bs))[::-1].tolist()
    xs = [np.random.randint(0, idim, l) for l in ilens]  # (bs, idim, x_len) luo0 => why not idim dimension
    ys = [np.random.randn(l, odim) for l in olens]
    ilens = torch.LongTensor(ilens).to(device)
    olens = torch.LongTensor(olens).to(device)
    xs = pad_list([torch.from_numpy(x).long() for x in xs], 0).to(device)
    ys = pad_list([torch.from_numpy(y).float() for y in ys], 0).to(device)
    labels = ys.new_zeros(ys.size(0), ys.size(1))  ## luo0 what it is used??
    for i, l in enumerate(olens):
        labels[i, l - 1:] = 1

    batch = {
        "xs": xs,
        "ilens": ilens,
        "ys": ys,
        "rs": ys,
        "olens": olens,
        "labels": labels
    }
    if spk_embed_dim is not None:
        spembs = torch.from_numpy(np.random.randn(bs, spk_embed_dim)).float().to(device)
        batch["spembs"] = spembs
    if spc_dim is not None:
        spcs = [np.random.randn(l, spc_dim) for l in olens]
        spcs = pad_list([torch.from_numpy(spc).float() for spc in spcs], 0).to(device)
        batch["extras"] = spcs
    """ Test code
    print("xs size:", xs.size())
    print("ys size:", ys.size())
    print("ilens:", ilens)
    print("olens:",olens)
    """
    return batch


@pytest.mark.parametrize(
    "model_dict, inference_dict", [
        ({}, {}),
        ({"use_masking": False}, {}),
        ({"bce_pos_weight": 10.0}, {}),
        # attention
        ({"atype": "forward"}, {}),
        ({"atype": "forward_ta"}, {}),
        # cbhg Test
        #({"use_cbhg": True},{}),
        # decoder
        # inference_dict
        ({},{"use_att_constraint": True})
    ])
def test_tacotron_gst(model_dict, inference_dict):
    # Set args
    model_args = make_tac2_gst_args(**model_dict)
    inference_args = make_tac2_gst_infer_args(**inference_dict)
    idim = 20
    maxin_len = 10
    odim = 10
    maxout_len = 20
    bs = 2

    # setup batch
    batch = prepare_inputs(bs, idim, odim, maxin_len, maxout_len,
                           model_args["spk_embed_dim"],
                           model_args["spc_dim"])
    # define model
    model = Tacotron2_GST(idim, odim, Namespace(**model_args))
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    loss = model(**batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # decodable
    model.eval()
    with torch.no_grad():
        outs, probs, attns = model.inference(batch["xs"][0], batch["rs"][0], Namespace(**inference_args))
        print("inference outs:", outs.size())
        #assert outs.size() == torch.Size([1, odim])   # ???
        #assert outs.size() == ()
        model.calculate_all_attentions(**batch)

    # Synthesizable


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")
@pytest.mark.parametrize(
    "model_dict, inference_dict", [
        ({}, {}),
        ({"use_masking": False}, {}),
        ({"bce_pos_weight": 10.0}, {})
    ])
def test_tacotron_gst_gpu(model_dict, inference_dict):
    pass
