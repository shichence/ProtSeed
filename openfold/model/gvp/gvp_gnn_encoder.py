# Copyright (c) Facebook, Inc. and its affiliates.
#
# Contents of this file were adapted from the open source fairseq repository.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from openfold.model.primitives import Linear, LayerNorm
from openfold.model.gvp.gvp_modules import SinusoidalPositionalEmbedding
from openfold.model.gvp.features import GVPInputFeaturizer, DihedralFeatures
from openfold.model.gvp.gvp_encoder import GVPEncoder
from openfold.model.gvp.util import nan_to_num
from openfold.utils.loss import check_inf_nan


class GVPGNN(nn.Module):
    """
    Transformer encoder consisting of *args.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(
        self,
        tf_dim,
        embed_dim,
        dropout,
        gvp_node_hidden_dim_scalar,
        gvp_node_hidden_dim_vector,
        gvp_edge_hidden_dim_scalar,
        gvp_edge_hidden_dim_vector,
        gvp_top_k_neighbors,
        gvp_num_encoder_layers,
        **kwargs,
    ):
        super(GVPGNN, self).__init__()
        self.tf_dim = tf_dim
        self.embed_dim = embed_dim

        # input layer
        # self.embed_scale = math.sqrt(embed_dim)
        # self.embed_positions = SinusoidalPositionalEmbedding(
        #     embed_dim=embed_dim,
        #     padding_idx=0,
        # )
        # self.embed_tf = Linear(tf_dim, embed_dim)
        # self.embed_gvp_input_features = Linear(15-9, embed_dim)
        # self.embed_dihedrals = DihedralFeatures(embed_dim)
        
        # gvp encoder
        self.gvp_encoder = GVPEncoder(
            gvp_node_hidden_dim_scalar,
            gvp_node_hidden_dim_vector,
            gvp_edge_hidden_dim_scalar,
            gvp_edge_hidden_dim_vector,
            gvp_top_k_neighbors,
            dropout,
            gvp_num_encoder_layers,
        )

        # output layer
        self.gvp_out_dim = gvp_node_hidden_dim_scalar
        self.embed_gvp_output = Linear(self.gvp_out_dim, embed_dim)

        # dropout
        self.dropout_module = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(embed_dim)

    def forward(
        self,
        tf: torch.Tensor,
        coords: torch.Tensor,
        seq_mask: torch.Tensor,
        coord_mask: torch.Tensor,
    ):
        """
        Args:
            tf: [*, N, 21] one-hot vector of amino acid types
            coords: [*, N, 3, 3] N, CA, C backbone coordinates 
            seq_mask: [*, N] sequence mask, 0 for padding
            coord_mask: [*, N, 3] 0 for padding
        """
        components = dict()
        coords = nan_to_num(coords)

        # [*, N]
        coord_mask = torch.all(coord_mask.bool(), dim=-1)
        padding_mask = ~(seq_mask.bool())

        # components["tokens"] = self.embed_tf(tf) * self.embed_scale
        # components["diherals"] = nan_to_num(
        #     self.embed_dihedrals(coords)
        # )

        # print(f'token dtype: {components["tokens"].dtype}')
        # check_inf_nan(components["tokens"])
        # print(f'diherals dtype: {components["diherals"].dtype}')
        # check_inf_nan(components["diherals"])
        # GVP encoder
        # [*, N, c_scalr], [*, N, c_vector, 3]
        gvp_out_scalars, gvp_out_vectors = self.gvp_encoder(coords,
                coord_mask, padding_mask)
        check_inf_nan(gvp_out_scalars)
        
        # gvp_out_vectors is not used in our model
        # but we hack the code for distributed training.
        gvp_out_vectors = torch.mean(gvp_out_vectors, dim=(-1, -2))
        gvp_out_scalars = gvp_out_scalars + gvp_out_vectors[..., None] * 0.0

        #print(f'gvp_scar dtype: {gvp_out_scalars.dtype}')
        components["gvp_out"] = self.embed_gvp_output(gvp_out_scalars)
        #print(f'gvp dtype: {components["gvp_out"].dtype}')
        check_inf_nan(components["gvp_out"])
        # In addition to GVP encoder outputs, also directly embed GVP input node
        # features to the Transformer
        # scalar_features, vector_features = GVPInputFeaturizer.get_node_features(
        #     coords, coord_mask, with_coord_mask=False)
        # components["gvp_input_features"] = self.embed_gvp_input_features(scalar_features)

        embed = sum(components.values())
        # for k, v in components.items():
        #     print(k, torch.mean(v, dim=(0,1)), torch.std(v, dim=(0,1)))

        x = embed
        check_inf_nan(x)
        # pos_embed = self.embed_positions(torch.argmax(tf, dim=-1)).to(x)
        # check_inf_nan(pos_embed)
        # x = x + pos_embed
        # check_inf_nan(x)
        x = self.dropout_module(x)
        check_inf_nan(x)
        x = self.layer_norm(x)
        check_inf_nan(x)
        return x


class GVPGNNEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(
        self,
        tf_dim,
        embed_dim,
        dropout,
        gvp_node_hidden_dim_scalar,
        gvp_node_hidden_dim_vector,
        gvp_edge_hidden_dim_scalar,
        gvp_edge_hidden_dim_vector,
        gvp_top_k_neighbors,
        gvp_num_encoder_layers,
        latent_dim,
        **kwargs,
    ):
        super(GVPGNNEncoder, self).__init__()
        self.gvp_gnn = GVPGNN(
            tf_dim,
            embed_dim,
            dropout,
            gvp_node_hidden_dim_scalar,
            gvp_node_hidden_dim_vector,
            gvp_edge_hidden_dim_scalar,
            gvp_edge_hidden_dim_vector,
            gvp_top_k_neighbors,
            gvp_num_encoder_layers,
        )
        self.linear_1 = Linear(embed_dim, embed_dim, init="default")
        self.linear_mean = Linear(embed_dim, latent_dim, init="default")
        self.linear_logvar = Linear(embed_dim, latent_dim, init="default")
        # self.layer_norm_mean = LayerNorm(latent_dim)
        # self.layer_norm_logvar = LayerNorm(latent_dim)
        self.relu = nn.ReLU()

    def forward(
        self,
        tf: torch.Tensor,
        coords: torch.Tensor,
        seq_mask: torch.Tensor,
        coord_mask: torch.Tensor,
    ):
        #print(f'encoder172, {coords.dtype}')
        # [*, N_res, c_gvp]
        gvp_gnn_emb = self.gvp_gnn(
            tf, coords,
            seq_mask, coord_mask,
        )
        check_inf_nan(gvp_gnn_emb)
        gvp_gnn_emb = gvp_gnn_emb * seq_mask[..., None]
        
        # [*, c_gvp]
        gvp_gnn_emb = torch.sum(gvp_gnn_emb, dim=-2)
        gvp_gnn_emb = gvp_gnn_emb / torch.sum(seq_mask, dim=-1, keepdim=True)

        # [*, c_l]
        latent_emb = self.relu(gvp_gnn_emb)
        latent_emb = self.linear_1(latent_emb)
        latent_emb = self.relu(latent_emb)

        latent_mean = self.linear_mean(latent_emb)
        latent_logvar = self.linear_logvar(latent_emb)

        #latent_mean = self.layer_norm_mean(latent_mean)
        #latent_logvar = self.layer_norm_logvar(latent_logvar)
        # [*, c_l]
        check_inf_nan(latent_mean)
        check_inf_nan(latent_logvar)
        return latent_mean, latent_logvar