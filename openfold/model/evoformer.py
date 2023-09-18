import torch
import torch.nn as nn
from typing import Tuple, Optional
from functools import partial

from openfold.model.primitives import Linear
from openfold.model.dropout import DropoutRowwise, DropoutColumnwise
from openfold.model.seq_update import SeqAttention, SeqTransition
from openfold.model.outer_product_mean import OuterProductMean
from openfold.model.pair_transition import PairTransition
from openfold.model.triangular_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from openfold.utils.checkpointing import checkpoint_blocks


class EvoformerBlock(nn.Module):
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_seq_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_seq: int,
        no_heads_pair: int,
        transition_n: int,
        seq_dropout: float,
        pair_dropout: float,
        inf: float,
    ):
        super(EvoformerBlock, self).__init__()
        # seq update
        self.seq_att = SeqAttention(
            c_m,
            c_hidden_seq_att,
            no_heads_seq,
            pair_bias=True,
            c_z=c_z,
            inf=inf,
        )
        self.seq_transition = SeqTransition(
            c_m=c_m,
            n=transition_n,
        )
        self.seq_dropout_layer = nn.Dropout(seq_dropout)

        # communication
        self.outer_product_mean = OuterProductMean(
            c_m,
            c_z,
            c_hidden_opm,
        )

        # pair update
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )
        self.tri_att_start = TriangleAttentionStartingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )
        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)
        self.ps_dropout_col_layer = DropoutColumnwise(pair_dropout)

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        seq_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_trans_mask = seq_mask if _mask_trans else None
        pair_trans_mask = pair_mask if _mask_trans else None

        # seq update
        m = m + self.seq_dropout_layer(
            self.seq_att(m, z, mask=seq_mask)
        )
        m = m + self.seq_transition(
            m,
            mask=seq_trans_mask,
            chunk_size=chunk_size,
        )

        # communication
        z = z + self.outer_product_mean(
            m,
            mask=seq_trans_mask,
            chunk_size=chunk_size,
        )

        # pair update
        z = z + self.ps_dropout_row_layer(self.tri_mul_out(z, mask=pair_mask))
        z = z + self.ps_dropout_row_layer(self.tri_mul_in(z, mask=pair_mask))
        z = z + self.ps_dropout_row_layer(
            self.tri_att_start(z, mask=pair_mask, chunk_size=chunk_size)
        )
        z = z + self.ps_dropout_col_layer(
            self.tri_att_end(z, mask=pair_mask, chunk_size=chunk_size)
        )
        z = z + self.pair_transition(
            z, mask=pair_trans_mask, chunk_size=chunk_size
        )

        return m, z


class EvoformerStack(nn.Module):
    """
    Main Evoformer trunk.

    Implements Algorithm 6.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_seq_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        c_s: int,
        no_heads_seq: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        seq_dropout: float,
        pair_dropout: float,
        inf: float,
        blocks_per_ckpt: int,
        clear_cache_between_blocks: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_m:
                Seq channel dimension
            c_z:
                Pair channel dimension
            c_hidden_seq_att:
                Hidden dimension in sequence attention
            c_hidden_opm:
                Hidden dimension in outer product mean module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            c_s:
                Channel dimension of the output "single" embedding
            no_heads_seq:
                Number of heads used for sequence attention
            no_heads_pair:
                Number of heads used for pair attention
            no_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the SequenceTransition
                hidden dimension
            seq_dropout:
                Dropout rate for sequence activations
            pair_dropout:
                Dropout used for pair activations
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
        """
        super(EvoformerStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList()

        for _ in range(no_blocks):
            block = EvoformerBlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_seq_att=c_hidden_seq_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_seq=no_heads_seq,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                seq_dropout=seq_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
            )
            self.blocks.append(block)

        self.linear = Linear(c_m, c_s)

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        seq_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            m:
                [*, N_res, C_m] sequence embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            seq_mask:
                [*, N_res] sequence mask
            pair_mask:
                [*, N_res, N_res] pair mask
        Returns:
            m:
                [*, N_res, C_m] sequence embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding
        """
        blocks = [
            partial(
                b,
                seq_mask=seq_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if self.clear_cache_between_blocks:
            def block_with_cache_clear(block, *args):
                # Releases all unoccupied cached memory
                # currently held by the caching allocator
                torch.cuda.empty_cache()
                return block(*args)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        m, z = checkpoint_blocks(
            blocks,
            args=(m, z),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
        )

        s = self.linear(m)

        return m, z, s
