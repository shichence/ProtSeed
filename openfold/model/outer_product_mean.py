# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from openfold.model.primitives import Linear
from openfold.utils.tensor_utils import chunk_layer


class OuterProductMean(nn.Module):
    """
    Adapted from Algorithm 10.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                seq embedding channel dimension
            c_z:
                pair embedding channel dimension
            c_hidden:
                hidden channel dimension
        """
        super(OuterProductMean, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z, init="final")

    def _opm(self, a, b):
        # a, b: [*, N_res, C]
        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bc,...de->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        return outer

    @torch.jit.ignore
    def _chunk(self, 
        a: torch.Tensor, 
        b: torch.Tensor, 
        chunk_size: int
    ) -> torch.Tensor:
        # a: [*, N_res, C]

        # Since the "batch dim" in this case is not a true batch dimension
        # (in that the shape of the output depends on it), we need to
        # iterate over it ourselves
        
        # [*, N_res, N_seq, C]
        a_reshape = a.reshape((-1,) + a.shape[-2:])
        b_reshape = b.reshape((-1,) + b.shape[-2:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size=chunk_size,
                no_batch_dims=1,
            )
            out.append(outer)
        outer = torch.stack(out, dim=0)
        outer = outer.reshape(a.shape[:-2] + outer.shape[1:])

        return outer

    def forward(self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_res, C_m] sequence embedding
            mask:
                [*, N_res] sequence mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, N_res, C_m]
        m = self.layer_norm(m)

        # [*, N_res, C]
        mask = mask.unsqueeze(-1)
        a = self.linear_1(m) * mask
        b = self.linear_2(m) * mask

        if chunk_size is not None:
            outer = self._chunk(a, b, chunk_size)
        else:
            outer = self._opm(a, b)

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...bc,...dc->...bdc", mask, mask)

        # [*, N_res, N_res, C_z]
        outer = outer / (self.eps + norm)

        return outer
