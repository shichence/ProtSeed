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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from openfold.model.primitives import Linear, LayerNorm, ipa_point_weights_init_
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_num,
)
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)
from openfold.utils.loss import check_inf_nan
from torch.distributions.categorical import Categorical

class ResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(ResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:

        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = ResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor, s_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        
        # FP16 friendly L2 norm computation
        current_scale = s.abs().max().detach().item()
        max_scale = (torch.finfo(s.dtype).max * 0.9) ** 0.5
        rescale = max(current_scale / max_scale, 1)
        
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum((s / rescale) ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        ) * rescale

        s = s / norm_denom

        return unnormalized_s, s


class SeqResnet(nn.Module):
    """
    Predict sequence type from single representation
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_types):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
        """
        super(SeqResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_types = no_types

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)
        self.linear_seq = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = ResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_types)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor,
        s_initial: torch.Tensor,
        s_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
            s_seq:
                [*, C_hidden] single embedding of the sequence type
        Returns:
            [*, no_types] predicted logits of sequence type
        """


        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)

        s_seq = self.relu(s_seq)
        s_seq = self.linear_seq(s_seq)

        s = s + s_initial + s_seq

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_types]
        s = self.linear_out(s)

        return s


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z)

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a = a * math.sqrt(1.0 / (3 * self.c_hidden))
        a = a + (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # As DeepMind explains, this manual matmul ensures that the operation
        # happens in float32.
        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (
                a[..., None, :, :, None] # [*, H, 1, N_res, N_res, 1]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :] # [*, H, 3, 1, N_res, P_v]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, N_res, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z.to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
            ).to(dtype=z.dtype)
        )

        return s


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector 
        """
        # [*, 6]
        update = self.linear(s)

        return update 


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):
    def __init__(self, c, num_layers, dropout_rate):
        super(StructureModuleTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class StructureModule(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_ipa,
        c_resnet,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        dropout_rate,
        no_blocks,
        no_transition_layers,
        no_resnet_blocks,
        no_angles,
        trans_scale_factor,
        epsilon,
        inf,
        denoise_enabled,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
        """
        super(StructureModule, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf
        self.denoise_enabled = denoise_enabled

        # To be lazily initialized later
        self.default_frames = None
        self.group_idx = None
        self.atom_mask = None
        self.lit_positions = None

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        self.bb_update = BackboneUpdate(self.c_s)

        self.angle_resnet = AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

        self.seq_emb_nn = nn.Sequential(
            Linear(restype_num + 1, self.c_s, init="relu"),
            nn.ReLU(),
            Linear(self.c_s, self.c_s, init="final"),
        )

        self.seq_resnet = SeqResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            restype_num + 1,
        )

    def forward(
        self,
        s,
        z,
        aatype,
        mask=None,
        initial_rigids=None,
        initial_seqs=None,
        denoise_feats=None,
    ):
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(z)

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        if initial_rigids is not None:
            rigids = Rigid.from_tensor_7(initial_rigids)
        else:
            rigids = Rigid.identity(
                s.shape[:-1],
                s.dtype,
                s.device,
                self.training,
                fmt="quat",
            )

        if initial_seqs is None:
            seqs = torch.zeros(
                (*s.shape[:-1], restype_num + 1),
                dtype=s.dtype,
                device=s.device,
            )
            seqs[..., -1] = 1.0
            # seqs.requires_grad_(self.training)
        else:
            seqs = initial_seqs
            check_inf_nan(seqs)

        outputs = []
        for i in range(self.no_blocks):
            # [*, N, C_s]
            seqs_emb = self.seq_emb_nn(seqs)
            check_inf_nan(seqs_emb)
            s = s + seqs_emb
            s = s + self.ipa(s, z, rigids, mask)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)
            check_inf_nan(self.bb_update.linear.weight)
            check_inf_nan(s)

            # [*, N]
            bb_update = self.bb_update(s)
            check_inf_nan(bb_update)

            if self.denoise_enabled:
                update_q_vec, update_t_vec = bb_update[..., :3], bb_update[..., 3:] # [*, N, 3]
                denoise_step_size = denoise_feats.get("denoise_step_size", None)
                used_sigmas_trans = denoise_feats.get("used_sigmas_trans", None)
                assert denoise_step_size is not None or used_sigmas_trans is not None
                # used_sigmas_rot = denoise_feats.get("used_sigmas_rot", None)

                ##########################
                # Rescale the translation update vector
                # The network is forced to learn the direction of the noise. i.e., `epsilon ~ N(0, I)`.
                ##########################
                if denoise_step_size is None: # training
                    assert used_sigmas_trans is not None
                    update_t_vec = update_t_vec * used_sigmas_trans # [*, N, 3]
                else: # inference time. Langevin dynamics
                    # step_noise = torch.randn_like(update_t_vec) * torch.sqrt(denoise_step_size * 2)
                    # step_noise = torch.randn_like(update_t_vec) * torch.sqrt(used_sigmas_trans * 2)
                    #update_t_vec = update_t_vec * denoise_step_size
                    update_t_vec = update_t_vec * used_sigmas_trans # [*, N, 3]

                ##########################
                # DECOUPLING THE ROTATION WILL CAUSE THE NETWORK TO DIVERGE!!
                ##########################
                """Quaternions and Rotations (https://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf)
                Given a quaternion, say, q = q_0 + \vq  = cos (\theta / 2) + \vu sin (\theta / 2)
                It represents a rotation of the vector through an angle \theta about \vu as the axis of rotation.
                
                  (cos(\theta), sin(\theta)q_1, sin(\theta)q_2, sin(\theta)q_3)
                ==(1          , tan(\theta)q_1, tan(\theta)q_2, tan(\theta)q_3).
                
                In AlphaFold, we predict the `update_q_vec`, which is a unnormalized vector q_ = (q_1, q_2, q_3).
                The norm of the q_ is thus tan(\theta/2), and (1/norm) * q_ is the rotation axis
                By rescaling the tan(\theta/2), we can rescale the predicted rotation.
                """
                ##########################
                # DELETED
                ##########################

                bb_update = torch.cat([update_q_vec, update_t_vec], dim=-1) # [*, N, 6]

            rigids = rigids.compose_q_update_vec(bb_update)

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones
            # here
            backb_to_global = Rigid(
                Rotation(
                    rot_mats=rigids.get_rots().get_rot_mats(), 
                    quats=None
                ),
                rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(
                self.trans_scale_factor
            )

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)
            check_inf_nan([unnormalized_angles, angles])

            # [*, N, 21]
            seqs_logits = self.seq_resnet(s, s_initial, seqs_emb)
            seqs = F.softmax(seqs_logits, dim=-1)
            check_inf_nan([seqs_logits, seqs])
            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            if not self.training or i == (self.no_blocks - 1):
                if not self.training:
                    # [*, N]
                    masked_seqs_logits = seqs_logits.clone()
                    masked_seqs_logits[..., -1] = -9999 # zero out UNK.
                    aatype_ = torch.argmax(masked_seqs_logits, dim=-1)
                    # dist = Categorical(logits=masked_seqs_logits)
                    # aatype_ = dist.sample()
                else:
                    aatype_ = aatype

                all_frames_to_global = self.torsion_angles_to_frames(
                    backb_to_global,
                    angles,
                    aatype_,
                ) # [*, N, 8]
                pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                    all_frames_to_global,
                    aatype_,
                ) # [*, N, 14, 3]
                all_frames_to_global = all_frames_to_global.to_tensor_4x4() # [*, N, 8, 4, 4]
                check_inf_nan([aatype_, pred_xyz])
            else:
                # Use dummy "all_frames_to_global" and "pred_xyz" to
                # save time if it is not the last step during the training.
                aatype_ = aatype
                batch_residue_dims = unnormalized_angles.shape[:-2] # [*, N]
                all_frames_to_global = torch.zeros(
                    *batch_residue_dims, 8, 4, 4,
                    dtype=unnormalized_angles.dtype,
                    device=unnormalized_angles.device,
                    requires_grad=self.training,
                ) # [*, N, 8, 4, 4]
                pred_xyz = torch.zeros(
                    *batch_residue_dims, 14, 3,
                    dtype=unnormalized_angles.dtype,
                    device=unnormalized_angles.device,
                    requires_grad=self.training,
                ) # [*, N, 14, 3]

            if i < (self.no_blocks - 1):
                rigids = rigids.stop_rot_gradient()
                # seqs = seqs.detach()

            preds = {
                "rigids": rigids.to_tensor_7(), # [*, N, 7]
                "frames": scaled_rigids.to_tensor_7(), # [*, N, 7]
                "unnormalized_angles": unnormalized_angles, # [*, N, 7, 2]
                "angles": angles, # [*, N, 7, 2]
                "singles": s, # [*, N, C_s]
                "sidechain_frames": all_frames_to_global, # [*, N, 8, 4, 4]
                "positions": pred_xyz, # [*, N, 14, 3]
                "seqs_logits": seqs_logits, # [*, N, 21]
                "seqs": seqs, # [*, N, 21]
                "aatype_": aatype_, # [*, N]
            }

            outputs.append(preds)


        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

    def _init_residue_constants(self, float_dtype, device):
        if self.default_frames is None:
            self.default_frames = torch.tensor(
                restype_rigid_group_default_frame,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.group_idx is None:
            self.group_idx = torch.tensor(
                restype_atom14_to_rigid_group,
                device=device,
                requires_grad=False,
            )
        if self.atom_mask is None:
            self.atom_mask = torch.tensor(
                restype_atom14_mask,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.lit_positions is None:
            self.lit_positions = torch.tensor(
                restype_atom14_rigid_group_positions,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
