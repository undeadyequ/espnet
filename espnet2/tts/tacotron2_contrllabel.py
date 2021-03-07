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
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.attentions import AttForward
from espnet.nets.pytorch_backend.rnn.attentions import AttForwardTA
from espnet.nets.pytorch_backend.rnn.attentions import AttLoc
from espnet.nets.pytorch_backend.tacotron2.decoder_extra_input import Decoder_extra_input
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.gst.style_encoder import StyleEncoder
from espnet2.tts.prosody.prosody_encoder import ProsodyEncoder
from espnet2.tts.ser.ser_model import DNNRevNetwork
from espnet2.tts.ser.inv_ser_combine import REVLSTMClassifier


class Tacotron2ControlLoss(torch.nn.Module):
    """Loss function module for Tacotron2."""

    def __init__(
        self, use_masking=True, use_weighted_masking=False, bce_pos_weight=20.0
    ):
        """Initialize Tactoron2 loss module.

        Args:
            use_masking (bool): Whether to apply masking
                for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to apply weighted masking in loss calculation.
            bce_pos_weight (float): Weight of positive sample of stop token.

        """
        super(Tacotron2ControlLoss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=torch.tensor(bce_pos_weight)
        )

        # NOTE(kan-bayashi): register pre hook function for the compatibility
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def forward(self, after_outs, before_outs, logits, ys, labels, olens, prosody, emfeats):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            logits (Tensor): Batch of stop logits (B, Lmax).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            labels (LongTensor): Batch of the sequences of stop token labels (B, Lmax).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Mean square error loss value.
            Tensor: Binary cross entropy loss value.

        """
        # make mask and apply it
        if self.use_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            ys = ys.masked_select(masks)
            after_outs = after_outs.masked_select(masks)
            before_outs = before_outs.masked_select(masks)
            labels = labels.masked_select(masks[:, :, 0])
            logits = logits.masked_select(masks[:, :, 0])

        # calculate loss
        l1_loss = self.l1_criterion(after_outs, ys) + self.l1_criterion(before_outs, ys)
        mse_loss = self.mse_criterion(after_outs, ys) + self.mse_criterion(
            before_outs, ys
        )
        bce_loss = self.bce_criterion(logits, labels)

        if emfeats.shape[1] > 8:
            audio_fts, txt_fts = emfeats[:, :8], emfeats[:, 8:]
        else:
            audio_fts = emfeats
        psd_loss = self.mse_criterion(prosody, audio_fts[:, :8])

        # make weighted mask and apply it
        if self.use_weighted_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            weights = masks.float() / masks.sum(dim=1, keepdim=True).float()
            out_weights = weights.div(ys.size(0) * ys.size(2))
            logit_weights = weights.div(ys.size(0))

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(masks).sum()
            mse_loss = mse_loss.mul(out_weights).masked_select(masks).sum()
            bce_loss = (
                bce_loss.mul(logit_weights.squeeze(-1))
                .masked_select(masks.squeeze(-1))
                .sum()
            )

        return l1_loss, mse_loss, bce_loss, psd_loss

    def _load_state_dict_pre_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Apply pre hook fucntion before loading state dict.

        From v.0.6.1 `bce_criterion.pos_weight` param is registered as a parameter but
        old models do not include it and as a result, it causes missing key error when
        loading old model parameter. This function solve the issue by adding param in
        state dict before loading as a pre hook function
        of the `load_state_dict` method.

        """
        key = prefix + "bce_criterion.pos_weight"
        if key not in state_dict:
            state_dict[key] = self.bce_criterion.pos_weight


class Tacotron2_controllable(AbsTTS):
    """Tacotron2_controllable module for end-to-end text-to-speech.

    This is a module of Spectrogram prediction network in Tacotron2 described
    in `Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_,
    which converts the sequence of characters into the sequence of Mel-filterbanks.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    Args:
        idim (int): Dimension of the inputs.
        odim: (int) Dimension of the outputs.
        spk_embed_dim (int, optional): Dimension of the speaker embedding.
        embed_dim (int, optional): Dimension of character embedding.
        elayers (int, optional): The number of encoder blstm layers.
        eunits (int, optional): The number of encoder blstm units.
        econv_layers (int, optional): The number of encoder conv layers.
        econv_filts (int, optional): The number of encoder conv filter size.
        econv_chans (int, optional): The number of encoder conv filter channels.
        dlayers (int, optional): The number of decoder lstm layers.
        dunits (int, optional): The number of decoder lstm units.
        prenet_layers (int, optional): The number of prenet layers.
        prenet_units (int, optional): The number of prenet units.
        postnet_layers (int, optional): The number of postnet layers.
        postnet_filts (int, optional): The number of postnet filter size.
        postnet_chans (int, optional): The number of postnet filter channels.
        output_activation (str, optional): The name of activation function for outputs.
        adim (int, optional): The number of dimension of mlp in attention.
        aconv_chans (int, optional): The number of attention conv filter channels.
        aconv_filts (int, optional): The number of attention conv filter size.
        cumulate_att_w (bool, optional): Whether to cumulate previous attention weight.
        use_batch_norm (bool, optional): Whether to use batch normalization.
        use_concate (bool, optional): Whether to concatenate encoder embedding with
            decoder lstm outputs.
        reduction_factor (int, optional): Reduction factor.
        spk_embed_dim (int, optional): Number of speaker embedding dimenstions.
        spk_embed_integration_type (str, optional): How to integrate speaker embedding.
        use_gst (str, optional): Whether to use global style token.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        gst_conv_layers (int, optional): The number of conv layers in GST.
        gst_conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in GST.
        gst_conv_kernel_size (int, optional): Kernal size of conv layers in GST.
        gst_conv_stride (int, optional): Stride size of conv layers in GST.
        gst_gru_layers (int, optional): The number of GRU layers in GST.
        gst_gru_units (int, optional): The number of GRU units in GST.
        dropout_rate (float, optional): Dropout rate.
        zoneout_rate (float, optional): Zoneout rate.
        use_masking (bool, optional): Whether to mask padded part in loss calculation.
        use_weighted_masking (bool, optional): Whether to apply weighted masking in
            loss calculation.
        bce_pos_weight (float, optional): Weight of positive sample of stop token
            (only for use_masking=True).
        loss_type (str, optional): How to calculate loss.
        use_guided_attn_loss (bool, optional): Whether to use guided attention loss.
        guided_attn_loss_sigma (float, optional): Sigma in guided attention loss.
        guided_attn_loss_lamdba (float, optional): Lambda in guided attention loss.

    """

    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        extra_idim: int = 8,
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
    ):
        """Initialize Tacotron2 module."""
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.extra_idim = extra_idim
        self.eos = idim - 1
        self.spk_embed_dim = spk_embed_dim
        self.cumulate_att_w = cumulate_att_w
        self.reduction_factor = reduction_factor
        self.use_gst = use_gst
        self.use_guided_attn_loss = use_guided_attn_loss
        self.loss_type = loss_type
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type

        # define activation function for the final output
        if output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, output_activation):
            self.output_activation_fn = getattr(F, output_activation)
        else:
            raise ValueError(
                f"there is no such an activation function. " f"({output_activation})"
            )

        # set padding idx
        padding_idx = 0
        self.padding_idx = padding_idx

        # define network modules
        self.enc = Encoder(
            idim=idim,
            embed_dim=embed_dim,
            elayers=elayers,
            eunits=eunits,
            econv_layers=econv_layers,
            econv_chans=econv_chans,
            econv_filts=econv_filts,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            dropout_rate=dropout_rate,
            padding_idx=padding_idx,
        )


        if self.use_gst:
            self.gst = StyleEncoder(
                idim=odim,  # the input is mel-spectrogram
                gst_tokens=gst_tokens,
                gst_token_dim=eunits,
                gst_heads=gst_heads,
                conv_layers=gst_conv_layers,
                conv_chans_list=gst_conv_chans_list,
                conv_kernel_size=gst_conv_kernel_size,
                conv_stride=gst_conv_stride,
                gru_layers=gst_gru_layers,
                gru_units=gst_gru_units,
            )
        # Prosody from text
        self.prosody = ProsodyEncoder(
                text_dim=embed_dim,
                prosody_dim=extra_idim,
                elayers=3,
                eunits=128
        )

        # Prosody from emo_labs and emo_feats(predicted prosody)
        #self.ser_rev = DNNRevNetwork(idim=8, edim=6)

        self.ser_rev = REVLSTMClassifier(input_dim=2301, emo_dim=4, prosody_dim=8)  # ck

        if spk_embed_dim is None:
            dec_idim = eunits
        elif spk_embed_integration_type == "concat":
            dec_idim = eunits + spk_embed_dim
        elif spk_embed_integration_type == "add":
            dec_idim = eunits
            self.projection = torch.nn.Linear(self.spk_embed_dim, eunits)
        else:
            raise ValueError(f"{spk_embed_integration_type} is not supported.")

        if atype == "location":
            att = AttLoc(dec_idim, dunits, adim, aconv_chans, aconv_filts)
        elif atype == "forward":
            att = AttForward(dec_idim, dunits, adim, aconv_chans, aconv_filts)
            if self.cumulate_att_w:
                logging.warning(
                    "cumulation of attention weights is disabled "
                    "in forward attention."
                )
                self.cumulate_att_w = False
        elif atype == "forward_ta":
            att = AttForwardTA(dec_idim, dunits, adim, aconv_chans, aconv_filts, odim)
            if self.cumulate_att_w:
                logging.warning(
                    "cumulation of attention weights is disabled "
                    "in forward attention."
                )
                self.cumulate_att_w = False
        else:
            raise NotImplementedError("Support only location or forward")
        self.dec = Decoder_extra_input(
            idim=dec_idim,
            odim=odim,
            extra_idim=self.extra_idim,
            att=att,
            dlayers=dlayers,
            dunits=dunits,
            prenet_layers=prenet_layers,
            prenet_units=prenet_units,
            postnet_layers=postnet_layers,
            postnet_chans=postnet_chans,
            postnet_filts=postnet_filts,
            output_activation_fn=self.output_activation_fn,
            cumulate_att_w=self.cumulate_att_w,
            use_batch_norm=use_batch_norm,
            use_concate=use_concate,
            dropout_rate=dropout_rate,
            zoneout_rate=zoneout_rate,
            reduction_factor=reduction_factor,
        )
        self.taco2_cotrl_loss = Tacotron2ControlLoss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
            bce_pos_weight=bce_pos_weight,
        )
        if self.use_guided_attn_loss:
            self.attn_loss = GuidedAttentionLoss(
                sigma=guided_attn_loss_sigma,
                alpha=guided_attn_loss_lambda,
            )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        emo_feats: torch.Tensor,
        emo_feats_lengths: torch.Tensor,
        spembs: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value.

        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        speech = speech[:, : speech_lengths.max()]  # for data-parallel

        batch_size = text.size(0)

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        ys = speech
        olens = speech_lengths

        eilens = emo_feats_lengths

        # make labels for stop prediction
        labels = make_pad_mask(olens - 1).to(ys.device, ys.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)

        # calculate tacotron2 outputs
        after_outs, before_outs, logits, att_ws, prosody = self._forward(
            xs, ilens, ys, olens, emo_feats, eilens, spembs
        )

        # modify mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_out = max(olens)
            ys = ys[:, :max_out]
            labels = labels[:, :max_out]
            labels[:, -1] = 1.0  # make sure at least one frame has 1

        # calculate taco2 loss
        l1_loss, mse_loss, bce_loss, psd_loss = self.taco2_cotrl_loss(
            after_outs, before_outs, logits, ys, labels, olens, prosody, emo_feats
        )
        if self.loss_type == "L1+L2":
            loss = l1_loss + mse_loss + bce_loss
        elif self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = mse_loss + bce_loss
        else:
            raise ValueError(f"unknown --loss-type {self.loss_type}")

        loss += psd_loss

        stats = dict(
            l1_loss=l1_loss.item(),
            mse_loss=mse_loss.item(),
            bce_loss=bce_loss.item(),
            psd_loss=psd_loss.item(),
        )

        # calculate attention loss
        if self.use_guided_attn_loss:
            # NOTE(kan-bayashi): length of output for auto-regressive
            # input will be changed when r > 1
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            attn_loss = self.attn_loss(att_ws, ilens, olens_in)
            loss = loss + attn_loss
            stats.update(attn_loss=attn_loss.item())

        stats.update(loss=loss.item())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        ys: torch.Tensor,
        olens: torch.Tensor,
        emo_feats: torch.Tensor,
        eilens: torch.Tensor,
        spembs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            xs:
            ilens:
            ys:
            olens:
            emo_feats:   audio_fts [,8] or audio_txt_fts [, 2301] (according to corp size)
            eilens:
            spembs:

        Returns:

        """
        hs, hlens = self.enc(xs, ilens)

        #prosody = self.prosody(hs)
        # Judge emo_fts is combined or audio_fts
        if emo_feats.shape[1] > 8:
            audio_fts, txt_fts = emo_feats[:, :8], emo_feats[:, 8:]
        else:
            audio_fts = emo_feats
        logging.info("emo_feats is {}".format(emo_feats.cpu().detach().numpy()))

        emo_labs, prosody = self.ser_rev(emo_feats)
        logging.info("emo_labs is {}".format(emo_labs.cpu().detach().numpy()))

        if self.use_gst:
            style_embs = self.gst(ys)
            hs = hs + style_embs.unsqueeze(1)
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)
        after_outs, before_outs, logits, att_ws = self.dec(hs, hlens, ys, audio_fts)
        return after_outs, before_outs, logits, att_ws, prosody

    def inference(
        self,
        text: torch.Tensor,
        speech: torch.Tensor = None,
        emo_feats: torch.Tensor = None,
        emolabs: torch.Tensor = None,
        optional_bias: torch.Tensor = None,
        spembs: torch.Tensor = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        use_teacher_forcing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            spembs (Tensor, optional): Speaker embedding vector (spk_embed_dim,).
            threshold (float, optional): Threshold in inference.
            minlenratio (float, optional): Minimum length ratio in inference.
            maxlenratio (float, optional): Maximum length ratio in inference.
            use_att_constraint (bool, optional): Whether to apply attention constraint.
            backward_window (int, optional): Backward window in attention constraint.
            forward_window (int, optional): Forward window in attention constraint.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing.

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Attention weights (L, T).

        """
        x = text
        y = speech
        spemb = spembs

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # inference with teacher forcing
        if use_teacher_forcing:
            assert speech is not None, "speech must be provided with teacher forcing."

            xs, ys = x.unsqueeze(0), y.unsqueeze(0)
            spembs = None if spemb is None else spemb.unsqueeze(0)
            ilens = x.new_tensor([xs.size(1)]).long()
            olens = y.new_tensor([ys.size(1)]).long()
            outs, _, _, att_ws = self._forward(xs, ilens, ys, olens, spembs)

            return outs[0], None, att_ws[0]

        # inference
        h = self.enc.inference(x)

        # Select controller (ref_audio has been extracted to emo_featsa in ESPnet)
        if emo_feats is not None:
            prosody = emo_feats.unsqueeze(0)                           # emo_feats controller(direct control)
        elif emolabs is not None:
            #_, prosody = self.ser_rev(emo_labs=emolabs.unsqueeze(0))                 # emo_labs controller
            _, prosody = self.ser_rev(emo_labs=emolabs.unsqueeze(0))   # ck

        else:
            print("Should input controller, emo_feats or emolabs!")
            prosody = self.ser_rev(emo_labs=[0, 1, 0, 0, 0, 0, 0])                        # if no controller

        if optional_bias is not None:       # is not used
            prosody += optional_bias

        if self.use_gst:
            style_emb = self.gst(y.unsqueeze(0))
            h = h + style_emb
        if self.spk_embed_dim is not None:
            hs, spembs = h.unsqueeze(0), spemb.unsqueeze(0)
            h = self._integrate_with_spk_embed(hs, spembs)[0]

        outs, probs, att_ws = self.dec.inference(
            h,
            extra=prosody,
            threshold=threshold,
            minlenratio=minlenratio,
            maxlenratio=maxlenratio,
            use_att_constraint=use_att_constraint,
            backward_window=backward_window,
            forward_window=forward_window,
        )

        return outs, probs, att_ws

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, eunits).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, eunits) if
                integration_type is "add" else (B, Tmax, eunits + spk_embed_dim).

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
        else:
            raise NotImplementedError("support only add or concat.")

        return hs
