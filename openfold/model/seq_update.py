import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from openfold.model.primitives import (
    Linear, 
    LayerNorm,
    Attention, 
)
from openfold.utils.tensor_utils import permute_final_dims, chunk_layer


class SeqTransition(nn.Module):
    """
    Feed-forward network applied to seq activations after attention.

    Adapted from Algorithm 9
    """
    def __init__(self, c_m, n):
        """
        Args:
            c_m:
                seq channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension
        """
        super(SeqTransition, self).__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m, init="final")

    def _transition(self, m, mask):
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    @torch.jit.ignore
    def _chunk(self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
         return chunk_layer(
             self._transition,
             {"m": m, "mask": mask},
             chunk_size=chunk_size,
             no_batch_dims=len(m.shape[:-2]),
         )


    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_res, C_m] seq activation
            mask:
                [*, N_res] seq mask
        Returns:
            m:
                [*, N_res, C_m] seq activation update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        m = self.layer_norm(m)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._transition(m, mask)

        return m


class SeqAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        pair_bias=False,
        c_z=None,
        inf=1e9,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super(SeqAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_in)

        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = LayerNorm(self.c_z)
            self.linear_z = Linear(
                self.c_z, self.no_heads, bias=False, init="normal"
            )
        
        self.mha = Attention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads
        )

    def _prep_inputs(self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # [*, N_res, C_m]
        m = self.layer_norm_m(m)

        if mask is None:
            # [*, N_res]
            mask = m.new_ones(
                m.shape[:-1],
            )

        # [*, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :]

        if (self.pair_bias and 
            z is not None and                       # For the 
            self.layer_norm_z is not None and       # benefit of
            self.linear_z is not None               # TorchScript
        ):
            # [*, N_res, N_res, C_z]
            z = self.layer_norm_z(z)
            
            # [*, N_res, N_res, no_heads]
            z = self.linear_z(z)
            
            # [*, no_heads, N_res, N_res]
            z = permute_final_dims(z, (2, 0, 1))

        return m, mask_bias, z

    def forward(self, 
        m: torch.Tensor, 
        z: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None, 
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_res, C_m] sequence embedding
            z:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_res] sequence mask
        """
        m, mask_bias, z = self._prep_inputs(m, z, mask)

        biases = [mask_bias]
        if z is not None:
            biases.append(z)

        m = self.mha(
            q_x=m, 
            kv_x=m, 
            biases=biases 
        )

        return m
