'''
Parameter

ref_enc_dims: dims of encoded reference audio
style_dims:   dims of reference audio style fo synthesis

'''

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from espnet.nets.pytorch_backend.emodetector.referenceEncoder import ReferenceEncoder


class GST(nn.Module):
    """
    Global Style Token model
    idim  : dimension of input (n_mels)
    ref_emb_dim  : out dimension from reference
    style_emb_dim:

    input : (N, T_y // r, n_mels * r)
    output: (N, 1, num_units)
    """
    def __init__(self,
                 idim,
                 ref_emb_dim=512,
                 style_emb_dim=512,
                 nums_head=8,
                 nums_token=10):
        super(GST, self).__init__()
        self.idim = idim
        self.ref_emb_dim = ref_emb_dim
        self.style_emb_dim = style_emb_dim

        self.ref_encoder = ReferenceEncoder(idim=self.idim,
                                            ref_emb_dim=self.ref_emb_dim)
        self.stl = STL(ref_emb_dim=self.ref_emb_dim,
                       style_emb_dim=self.style_emb_dim,
                       nums_head=nums_head,
                       nums_token=nums_token)

    def forward(self, ref_mel):
        ref_hid = self.ref_encoder(ref_mel)  # (N, E//2)
        style_emb = self.stl(ref_hid)
        return style_emb

    def inference(self, ref_mel):
        assert len(ref_mel.size()) == 2
        ref_mel_c = ref_mel.unsqueeze(0)
        return self.forward(ref_mel_c)[0]


class STL(nn.Module):
    """
    Style Token layer
    Got style from reference
    input:
        ref_emb_dim   :  reference dimension
        style_emb_dim :  Style dimension
        nums_head     :  How is the meaningful diverse of the dimension
        nums_token    :  T_K, equals to the numbers of style type
    output:
        style    (N, E)
    """
    def __init__(self,
                 ref_emb_dim=512,
                 style_emb_dim=512,
                 nums_head=8,
                 nums_token=10
                 ):
        super(STL, self).__init__()
        # Store token parameter  why set to ref // nums_head
        self.tokens = nn.Parameter(torch.FloatTensor(nums_token, ref_emb_dim // nums_head))  # (nums_token, ref//head??)

        d_q = ref_emb_dim            # Query dim
        d_k = ref_emb_dim // nums_head
        self.mh_att = MultiHeadAttention(d_q, d_k, style_emb_dim, nums_head)

        init.normal_(self.tokens, mean=0, std=0.5)

    def forward(self, ref_hid):
        N = ref_hid.size(0)
        ref_hid = ref_hid.unsqueeze(1)      # [N, 1, E]
        keys = F.tanh(self.tokens).unsqueeze(0).expand(N, -1, -1)  # (N, n_token, E//n_head)
        style_emb = self.mh_att(ref_hid, keys) # [N, 1, E]
        return style_emb


class MultiHeadAttention(nn.Module):
    """
    input
        query: [N, T_q, d_q]  (hidden vector)  (T_q = 1)
        key  : [N, T_k, d_k]  (Token)          (T_k = n_token)
        value: same as key  # omitted
        nums_units : Linear-transformed dimension, as well as attention dimension
        nums_heads :
    output
        attention(q, k, v) : [N, T_q, num_heads]
    other:
        att_score(q, k): [N, T_q, T_k]
    """
    def __init__(self,
                 d_q,
                 d_k,
                 num_units=512,
                 num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_q = d_q
        self.num_units = num_units

        self.query = nn.Linear(d_q, num_units)
        self.key = nn.Linear(d_k, num_units)
        self.value = nn.Linear(d_k, num_units)

    def forward(self, q, k):
        q = self.query(q)   # (N, T_q, num_units)
        v = self.value(k)     # (N, T_k, num_units)
        k = self.key(k)   # (N, T_k, num_units)

        split_size = self.num_units // self.num_heads  # Split_size
        q_split = torch.stack(torch.split(q, split_size, dim=2), dim=0)  # (h, N, T_q, split_size)
        k_split = torch.stack(torch.split(k, split_size, dim=2), dim=0)  # (h, N, T_k, split_size)
        v_split = torch.stack(torch.split(v, split_size, dim=2), dim=0)  # (h, N, T_k, split_size)

        # Scaled dot-product attention
        score = F.softmax(torch.matmul(q_split, k_split.transpose(2,3)) / (self.d_k**0.5))   # (h, N, T_q, T_k)
        ## Mask ???
        att = torch.matmul(score, v_split)   # (h, N, T_q, split_size)
        att = torch.cat(torch.split(att, 1, dim=0), dim=3)  # (N, T_q, nums_units)
        att = att.squeeze(0)  # (N, T_q, nums_units)
        return att