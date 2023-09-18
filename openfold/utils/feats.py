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
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch_scatter

from openfold.np import protein
import openfold.np.residue_constants as rc
from openfold.utils.loss import check_inf_nan
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import (
    batched_gather,
    permute_final_dims,
    one_hot,
    tree_map,
    tensor_tree_map,
)


def compute_contact_ca(
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 10.0,
    eps: float = 1e-10,
    deterministic: bool = False,
    block: bool = False,
    droploop: bool = True,
    ss_feat: Optional[torch.Tensor] = None,
    is_training: bool = False,
) -> torch.Tensor:
    """
        Compute the contact map of alpha carbons

        Args:
            deterministic:
                If True, contact = 1 for pairs with distance < cutoff
                If False, contact ~ Bernoulli(f(distance))
        Returns:
            [*, N, N] binary contact map
    """
    if block and not deterministic:
        raise ValueError(
            "blockwise contact and random contact can not be True simultanesouly."
        )
    if block and ss_feat is None:
        raise ValueError(
            "second structure block index should be given when block is True."
        )
    ca_pos = rc.atom_order["CA"]

    # [*, N, 3]
    all_atom_positions = all_atom_positions[..., ca_pos, :]

    # [*, N, 1]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    # if not is_training:
    #     # all_atom_positions += torch.randn_like(all_atom_positions) * 5
    #     all_atom_mask_2 = torch.randint_like(all_atom_mask, 0, 2)
    #     all_atom_mask = all_atom_mask * all_atom_mask_2

    dmat = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    
    # [*, N, N]
    contact = (
        (dmat < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
    ).type(torch.long)

    if not deterministic:
        p = 0.5 * (torch.cos(dmat * math.pi / cutoff) + 1.0)
        contact_mask = torch.bernoulli(p).type(contact.dtype)
        contact = contact * contact_mask

    if block:
        # [*, N, N]
        if contact.dim() != 3:
            raise ValueError(
                f"blockwise contact is only supported when contact.dim() == 3\n"
                f"found tensor with shape {contact.size()}"
            )

        # compute ss_index
        # [*, N]
        ss_feat = torch.argmax(ss_feat, dim=-1)
        ss_index = torch.cat(
            (
                torch.zeros_like(ss_feat[..., :1]),
                torch.cumsum(ss_feat[..., :-1] != ss_feat[..., 1:], dim=-1)
            ),
            dim=-1,
        ).to(dtype=contact.dtype)
        block_size = ss_index.max() + 1
        
        pair2block_id = ss_index[..., None] * block_size + ss_index[..., None, :] # [*, N, N]
        pair2block_id_ = pair2block_id.flatten(-2, -1) # [*, N * N]
        contact_ = contact.flatten(-2, -1) # [*, N * N]

        # [*, b_size * b_size]
        block_sum_ = torch_scatter.scatter_add(
            contact_, pair2block_id_, dim=-1, dim_size=block_size**2
        )
        contact_blockwise_ = torch.gather(block_sum_, -1, pair2block_id_) # [*, N * N]
        contact_blockwise = contact_blockwise_.reshape(-1, *contact.shape[-2:]) # [*, N, N]

        contact = (
            (contact_blockwise > 0)
            * all_atom_mask
            * permute_final_dims(all_atom_mask, (1, 0))
        ).type(torch.long)
        if droploop:
            # discard all information about loop - non_loop block.
            ss_mask1 = ss_feat != rc.second_structures_order['C']
            ss_mask1 = ss_mask1.to(dtype=all_atom_mask.dtype)
            ss_mask1 = ss_mask1[..., None] * ss_mask1[..., None, :]
            ss_mask2 = ss_feat == rc.second_structures_order['C']
            ss_mask2 = ss_mask2.to(dtype=all_atom_mask.dtype)
            ss_mask2 = ss_mask2[..., None] * ss_mask2[..., None, :]
            ss_mask = ((ss_mask1 + ss_mask2) > 0).type(torch.long)
            contact = contact * ss_mask

    # if not is_training:
    #     torch.set_printoptions(profile="full")
    #     ss_feat = torch.argmax(ss_feat, dim=-1)
    #     print(contact[0][:20,:20])
    #     torch.set_printoptions(profile="default")
    return contact


def compute_dist_ca(
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    eps: float = 1e-8,
    cutoff: float = 20.0,
) -> torch.Tensor:
    """
        Compute the contact map of alpha carbons

        Args:
            deterministic:
                If True, contact = 1 for pairs with distance < cutoff
                If False, contact ~ Bernoulli(f(distance))
        Returns:
            [*, N, N] binary contact map
    """

    ca_pos = rc.atom_order["CA"]

    # [*, N, 3]
    all_atom_positions = all_atom_positions[..., ca_pos, :]

    # [*, N, 1]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    dmat = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # [*, N, N]
    valid_pair_mask = all_atom_mask * permute_final_dims(all_atom_mask, (1, 0))
    dmat = torch.where(dmat < cutoff, dmat, cutoff * torch.ones_like(dmat))
    dmat = torch.where(valid_pair_mask > 0, dmat, cutoff * torch.ones_like(dmat))

    return dmat


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    is_gly = aatype == rc.restype_order["G"]
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    pseudo_beta = torch.where(
        is_gly[..., None].expand(*((-1,) * len(is_gly.shape)), 3),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_masks[..., ca_idx],
            all_atom_masks[..., cb_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


def atom14_to_atom37(atom14, batch):
    atom37_data = batched_gather(
        atom14,
        batch["residx_atom37_to_atom14"],
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )

    atom37_data = atom37_data * batch["atom37_atom_exists"][..., None]

    return atom37_data


def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
):
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat(
        [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
    )

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 14, 4, 4]
    default_4x4 = default_frames[aatype, ...]

    # [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions
