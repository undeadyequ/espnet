# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron 2 related modules for ESPnet2."""

import logging
from typing import Dict
from typing import Sequence
from typing import Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import GuidedAttentionLoss
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import Tacotron2Loss
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.attentions import AttForward
from espnet.nets.pytorch_backend.rnn.attentions import AttForwardTA
from espnet.nets.pytorch_backend.rnn.attentions import AttLoc
from espnet.nets.pytorch_backend.tacotron2.decoder import Decoder
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.gst.style_encoder import StyleEncoder
from espnet2.tts.ct.contrl_encoder import ContrlEncoder


class ControllableEmoTTS(Tacotron2):
    def __init__(self,
                 # network structure related
                 idim: int,
                 odim: int,
                 embed_dim: int = 512,
                 elayers: int = 1,
                 eunits: int = 512,
                 econv_layers: int = 3,
                 econv_chans: int = 512,
                 econv_filts: int = 5,
                 atype: str = "location",
                 adim: int = 512,
                 aconv_chans: int = 32,
                 aconv_filts: int = 15,
                 cumulate_att_w: bool = True,
                 dlayers: int = 2,
                 dunits: int = 1024,
                 prenet_layers: int = 2,
                 prenet_units: int = 256,
                 postnet_layers: int = 5,
                 postnet_chans: int = 512,
                 postnet_filts: int = 5,
                 output_activation: str = None,
                 use_batch_norm: bool = True,
                 use_concate: bool = True,
                 use_residual: bool = False,
                 reduction_factor: int = 1,
                 spk_embed_dim: int = None,
                 spk_embed_integration_type: str = "concat",
                 use_gst: bool = False,
                 gst_tokens: int = 10,
                 gst_heads: int = 4,
                 gst_conv_layers: int = 6,
                 gst_conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
                 gst_conv_kernel_size: int = 3,
                 gst_conv_stride: int = 2,
                 gst_gru_layers: int = 1,
                 gst_gru_units: int = 128,
                 # training related
                 dropout_rate: float = 0.5,
                 zoneout_rate: float = 0.1,
                 use_masking: bool = True,
                 use_weighted_masking: bool = False,
                 bce_pos_weight: float = 5.0,
                 loss_type: str = "L1+L2",
                 use_guided_attn_loss: bool = True,
                 guided_attn_loss_sigma: float = 0.4,
                 guided_attn_loss_lambda: float = 1.0,
                 emo_distrb_dim: int = 5,
                 style_token_dim: int = 6
                 ):

        super().__init__(idim,
                         odim,
                         embed_dim,
                         elayers,
                         eunits,
                         econv_layers,
                         econv_chans,
                         econv_filts,
                         atype,
                         adim,
                         aconv_chans,
                         aconv_filts,
                         cumulate_att_w,
                         dlayers,
                         dunits,
                         prenet_layers,
                         prenet_units,
                         postnet_layers,
                         postnet_chans,
                         postnet_filts,
                         output_activation,
                         use_batch_norm,
                         use_concate,
                         use_residual,
                         reduction_factor,
                         spk_embed_dim,
                         spk_embed_integration_type,
                         use_gst,
                         gst_tokens,
                         gst_heads,
                         gst_conv_layers,
                         gst_conv_chans_list,
                         gst_conv_kernel_size,
                         gst_conv_stride,
                         gst_gru_layers,
                         gst_gru_units,
                         # training related
                         dropout_rate,
                         zoneout_rate,
                         use_masking,
                         use_weighted_masking,
                         bce_pos_weight,
                         loss_type,
                         use_guided_attn_loss,
                         guided_attn_loss_sigma,
                         guided_attn_loss_lambda)
        self.prosody_encoder = torch.nn.Linear(emo_distrb_dim, style_token_dim)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spembs: torch.Tensor = None,
        emo_distrbs: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        style_token = self.prosody_encoder([text, emo_distrbs])
        text_style = text + style_token

        super(ControllableEmoTTS, self).forward(
            text_style,
            text_lengths,
            speech,
            speech_lengths,
            spembs)

    def inference(
        self,
        text: torch.Tensor,
        speech: torch.Tensor = None,
        spembs: torch.Tensor = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        use_teacher_forcing: bool = False,
        emo_distrbs : torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        style_token = self.prosody_encoder([text, emo_distrbs])
        text_style = text + style_token

        super(ControllableEmoTTS, self).inference(
            text_style,
            speech,
            spembs,
            threshold,
            minlenratio,
            maxlenratio,
            use_att_constraint,
            backward_window,
            forward_window,
            use_teacher_forcing)
