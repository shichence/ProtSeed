# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold.model.gvp.features import GVPGraphEmbedding
from openfold.model.gvp.gvp_modules import GVPConvLayer
from openfold.model.gvp.gvp_utils import unflatten_graph


class GVPEncoder(nn.Module):

    def __init__(
        self,
        node_hidden_dim_scalar,
        node_hidden_dim_vector,
        edge_hidden_dim_scalar,
        edge_hidden_dim_vector,
        top_k_neighbors,
        dropout,
        num_encoder_layers,
    ):
        super().__init__()
        self.embed_graph = GVPGraphEmbedding(
            top_k_neighbors,
            node_hidden_dim_scalar,
            node_hidden_dim_vector,
            edge_hidden_dim_scalar,
            edge_hidden_dim_vector,
        )

        node_hidden_dim = (
            node_hidden_dim_scalar,
            node_hidden_dim_vector,
        )
        edge_hidden_dim = (
            edge_hidden_dim_scalar,
            edge_hidden_dim_vector,
        )
        
        conv_activations = (F.relu, torch.sigmoid)
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(
                node_hidden_dim,
                edge_hidden_dim,
                drop_rate=dropout,
                vector_gate=True,
                attention_heads=0,
                n_message=3,
                conv_activations=conv_activations,
                n_edge_gvps=0,
                eps=1e-4,
                layernorm=True,
            )
            for i in range(num_encoder_layers)
        )

    def forward(self, coords, coord_mask, padding_mask):
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(
                coords, coord_mask, padding_mask)
        for i, layer in enumerate(self.encoder_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings,
                    edge_index, edge_embeddings)

        node_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
        return node_embeddings
