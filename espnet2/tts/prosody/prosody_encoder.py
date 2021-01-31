from typeguard import check_argument_types
from typing import Sequence
import torch
import six


class ProsodyEncoder(torch.nn.Module):
    """
    This module is prosody encoder introduced in "Controllable neural text-to-speech synthesis using
    intuitive prosodic features": "https://arxiv.org/abs/2009.06775"

    """

    def __init__(
            self,
            text_dim: int = 512,
            prosody_dim: int = 8,
            elayers=3,
            eunits=128
    ):
        """

        :param idim:
        :param prosody_dim:
        :param elayers:
        :param eunits:
        """
        super(ProsodyEncoder, self).__init__()
        self.eunits = eunits
        self.elayers = elayers
        self.prosody_dim = prosody_dim
        self.lstms = torch.nn.LSTM(text_dim, eunits, elayers, batch_first=True)
        self.lns = torch.nn.ModuleList()
        self.lns += [
            torch.nn.Sequential(
                torch.nn.Linear(eunits, prosody_dim),
                torch.nn.Tanh()
            )]

        """
        self.lstms = torch.nn.ModuleList()

        for layer in six.moves.range(elayers):
            iunits = idim if layer == 0 else eunits
            lstm = torch.nn.LSTMCell(iunits, eunits)
            self.lstms += lstm
        self.prosody_fc = torch.nn.Linear(eunits, prosody_dim)

        """

    def forward(self, text_hs):
        """

        Args:
            text_hs: [ t_l, t_units ]

        Returns:

        """
        out, (h_n, c_n) = self.lstms(text_hs)
        prosody = h_n[-1]    # h_s of last layer
        for i in range(len(self.lns)):
            prosody = self.lns[i](prosody)
        return prosody

    def inference(self, text_hs):
        """

        Args:
            text_hs:

        Returns:
            prosody:
        """
        text_hs = text_hs.unsqueeze(0)
        out, (h_n, c_n) = self.lstms(text_hs)
        prosody = h_n[-1]       # h_s of last layer
        for i in range(len(self.lns)):
            prosody = self.lns[i](prosody)
        return prosody



"""
    def _zero_state(self, text_hs: torch.Tensor):
        return text_hs.new_zeros(text_hs.size(0), self.lstms[0].hidden_size)


    def forward(self, text_hs, th_lens, emofeats):
        z_list = [self._zero_state(text_hs)]
        c_list = [self._zero_state(text_hs)]
        for _ in six.moves.range(1, len(self.lstms)):
            z_list += [self._zero_state(text_hs)]
            c_list += [self._zero_state(text_hs)]


        prev_out = text_hs.new_zeros(text_hs.size(0), self.prosody_dim)

        for ef in emofeats:
            z_list[0], c_list[0] = self.lstms[0](prev_out, (z_list[0], c_list[0]))
            for i in range(1, self.layers):
                z_list[i], c_list[i] = self.lstms[i](z_list[i-1], (z_list[i], c_list[i]))

            prev_out = ef        # teacher force
            lstm_out = z_list[-1]
            prosody = self.prosody_fc(lstm_out)
            return prosody

    def inference(self, text_hs):
        with True:
            z_list, c_list = self.lstms(text_hs)
            prosody = self.prosody_out(z_list)
            return prosody
"""