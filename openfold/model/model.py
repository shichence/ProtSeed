import math
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from openfold.model.primitives import Linear

from openfold.utils.feats import (
    pseudo_beta_fn,
    atom14_to_atom37,
)
from openfold.model.embedders import (
    InputEmbedder,
    RecyclingEmbedder,
    Ca_Aware_Embedder,
)
from openfold.model.evoformer import EvoformerStack
from openfold.model.heads import AuxiliaryHeads
import openfold.np.residue_constants as residue_constants
from openfold.model.structure_module import StructureModule
from openfold.model.gvp.gvp_gnn_encoder import GVPGNNEncoder
from openfold.model.conv import DistanceMapEncoder, DistanceMapDecoder
from openfold.utils.loss import check_inf_nan
from openfold.utils.feats import compute_contact_ca, compute_dist_ca
from openfold.utils.tensor_utils import tensor_tree_map, permute_final_dims
from openfold.utils.rigid_utils import Rotation, Rigid


class AlphaFold(nn.Module):
    """
    Alphafold 2.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(AlphaFold, self).__init__()

        self.globals = config.globals
        self.config = config.model

        # Main trunk + structure module
        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"],
        )
        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )

        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )
        self.structure_module = StructureModule(
            **self.config["structure_module"],
        )
        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )

        if self.config.latent_enabled:
            self.gvp_gnn_enc = GVPGNNEncoder(
                **self.config["gvp_gnn_enc"]
            )
            self.linear_latent_s = Linear(
                self.globals.c_l,
                self.globals.c_s,
            )

        if self.config.cnn_enabled:
            assert not self.config.latent_enabled
            assert not self.config.contact_enabled
            self.dist_enc = DistanceMapEncoder(**self.config["cnn"])
            self.dist_dec = DistanceMapDecoder(**self.config["cnn"])

        if self.config.denoise_enabled:
            # log linear, i.e. exponential weight for larger noise (begin > end)
            sigmas_trans = torch.tensor(
                np.exp(np.linspace(np.log(self.config.denoise.sigma_begin_trans), np.log(self.config.denoise.sigma_end_trans),
                                self.config.denoise.num_noise_level)), dtype=torch.float32) # convert type later
            sigmas_rot = torch.tensor(
                np.exp(np.linspace(np.log(self.config.denoise.sigma_begin_rot), np.log(self.config.denoise.sigma_end_rot),
                                self.config.denoise.num_noise_level)), dtype=torch.float32) # convert type later
            #self.register_buffer("sigmas_trans", sigmas_trans)
            #self.register_buffer("sigmas_rot", sigmas_rot)
            self.sigmas_trans = nn.Parameter(sigmas_trans, requires_grad=False) # (num_noise_level)
            self.sigmas_rot = nn.Parameter(sigmas_rot, requires_grad=False) # (num_noise_level)
            # self.ca_aware_embedder = Ca_Aware_Embedder(
            #     **config["recycling_embedder"],
            # )

    def compute_latent(
        self,
        tf,
        all_atom_positions,
        all_atom_mask,
        seq_mask,
    ):
        if not self.config.latent_enabled:
            raise ValueError(
                "compute_laatent should not be called when latent is not enabled."
            )

        coords_index = [
            residue_constants.atom_order["N"],
            residue_constants.atom_order["CA"],
            residue_constants.atom_order["C"],
        ]
        coords_index = tf.new_tensor(coords_index).to(dtype=torch.long)

        # [*, N, 3, 3]
        all_atom_positions = all_atom_positions[..., coords_index, :]
        # [*, N, 3]
        all_atom_mask = all_atom_mask[..., coords_index]

        # [*, c_l]
        latent_mean, latent_logvar = self.gvp_gnn_enc(
            tf,
            all_atom_positions,
            seq_mask,
            all_atom_mask,
        )
        return latent_mean, latent_logvar

    def iteration(
        self, feats, m_1_prev, z_prev, x_prev, seqs_prev,
        initial_rigids=None,
        initial_seqs=None,
        _recycle=True,
        denoise_feats=None,
    ):
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        # Ignore it if you are using FP32.
        dtype = next(self.parameters()).dtype
        #print('model dtype', dtype)
        for k in feats:
            #print(feats[k].dtype)
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        n = feats["target_feat"].shape[-2]

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        check_inf_nan(feats["all_atom_positions"])

        ## Calculate contact
        ## [*, N, N]
        contact = None
        if self.config.contact_enabled:
            contact = compute_contact_ca(
                feats["all_atom_positions"],
                feats["all_atom_mask"],
                ss_feat=feats["ss_feat"],
                is_training=self.training,
                **self.config["contact"],
            )
            check_inf_nan(contact)
        if "fold_contact" in feats:
            contact = feats["fold_contact"]
        # Initialize the seq and pair representations
        # m: [*, N, C_m]
        # z: [*, N, N, C_z]
        m, z = self.input_embedder(
            #feats["target_feat"],
            feats["ss_feat"],
            feats["residue_index"],
            contact=contact,
        )

        if self.config.cnn_enabled:
            # [*, N, N]
            ca_dist = compute_dist_ca(
                feats["all_atom_positions"],
                feats["all_atom_mask"],
                cutoff=20.0,
                )
            # [*, 1, N, N]
            ca_dist = ca_dist[..., None, :, :]
            
            # [*, nz]
            latent_mean, latent_logvar = self.dist_enc(ca_dist)
            outputs["latent_mean"] = latent_mean
            outputs["latent_logvar"] = latent_logvar
            check_inf_nan([latent_mean, latent_logvar])
            latent = torch.randn(
                latent_mean.size(),
                dtype=latent_mean.dtype,
                device=latent_mean.device,
            )
            if self.training:
                latent = latent_mean + latent * torch.exp(0.5 * latent_logvar)
            
            # [*, nz, 1, 1]
            latent = latent[..., None, None]
            # [*, c_z, 256, 256]
            dec_ca_dist = self.dist_dec(latent)
            # [*, 256, 256, c_z]
            dec_ca_dist = permute_final_dims(dec_ca_dist, (1, 2, 0))
            z = z + dec_ca_dist.to(dtype=z.dtype)
            outputs["latent_pair"] = z

        check_inf_nan([m,z])
        # Initialize the recycling embeddings, if needs be
        if None in [m_1_prev, z_prev]:
            # [*, N, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N, N, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n, n, self.config.input_embedder.c_z),
                requires_grad=False,
            )

        if x_prev is None:
            # [*, N, 37, 3]
            x_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )

        if seqs_prev is None:
            # [*, N, 21]
            seqs_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.restype_num + 1),
                requires_grad=False,
            )
            seqs_prev[..., -1] = 1.0

        # [*, N, 3]
        x_prev = pseudo_beta_fn(
            feats["aatype"], x_prev, None
        ).to(z.dtype)

        # m_1_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            x_prev,
            seqs_prev,
        )
        check_inf_nan([m_1_prev_emb, z_prev_emb])
        # If the number of recycling iterations is 0, skip recycling
        # altogether. We zero them this way instead of computing them
        # conditionally to avoid leaving parameters unused, which has annoying
        # implications for DDP training.
        if not _recycle:
            m_1_prev_emb = m_1_prev_emb * 0
            z_prev_emb = z_prev_emb * 0

        # if self.config.denoise_enabled:
        #     assert initial_rigids is not None
        #     x_ca = initial_rigids[..., 4:].detach() * self.config.structure_module.trans_scale_factor # [*, N_res, 3]
        #     z_update = self.ca_aware_embedder(x_ca) # [*, N_res, N_res, C_z]
        #     z = z + z_update

        # [*, N, C_m]
        m = m + m_1_prev_emb

        # [*, N, N, C_z]
        z = z + z_prev_emb

        # Possibly prevents memory fragmentation
        del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb


        # Run sequence + pair embeddings through the trunk of the network
        # m: [*, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]
        m, z, s = self.evoformer(
            m,
            z,
            seq_mask=seq_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            chunk_size=self.globals.chunk_size,
            _mask_trans=self.config._mask_trans,
        )
        check_inf_nan([m, z, s])
        # print('m,z, s', m.dtype, z.dtype, s.dtype)

        outputs["pair"] = z
        outputs["single"] = s

        ## Compute latent
        ## [*, c_l]
        if self.config.latent_enabled:
            latent_mean, latent_logvar = self.compute_latent(
                feats["target_feat"],
                feats["all_atom_positions"],
                feats["all_atom_mask"],
                seq_mask,
            )
            outputs["latent_mean"] = latent_mean
            outputs["latent_logvar"] = latent_logvar
            check_inf_nan([latent_mean, latent_logvar])
            latent = torch.randn(
                latent_mean.size(),
                dtype=latent_mean.dtype,
                device=latent_mean.device,
            )
            check_inf_nan(latent)
            if self.training:
                latent = latent_mean + latent * torch.exp(0.5 * latent_logvar)
            check_inf_nan(latent)

            # latent: [*, c_l]
            # s: [*, N, c_s]
            latent = self.linear_latent_s(latent)[..., None, :]
            s = s + latent

        # Predict 3D structure
        outputs["sm"] = self.structure_module(
            s,
            z,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            initial_rigids=initial_rigids,
            initial_seqs=initial_seqs,
            denoise_feats=denoise_feats,
        )

        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]
        outputs["final_aatype"] = outputs["sm"]["aatype_"][-1]
        outputs["final_seqs"] = outputs["sm"]["seqs"][-1]
        outputs["final_atom_positions_atom14"] = outputs["sm"]["positions"][-1]
        # Save embeddings for use during the next recycling iteration

        # [*, N, C_m]
        m_1_prev = m

        # [*, N, N, C_z]
        z_prev = z

        # [*, N, 37, 3]
        x_prev = outputs["final_atom_positions"]
        
        seqs_prev = outputs["sm"]["seqs"][-1]

        return outputs, m_1_prev, z_prev, x_prev, seqs_prev

    def _disable_activation_checkpointing(self):
        self.evoformer.blocks_per_ckpt = None

    def _enable_activation_checkpointing(self):
        self.evoformer.blocks_per_ckpt = (
            self.config.evoformer_stack.blocks_per_ckpt
        )

    def forward(self, batch):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        seqs_prev = None
        
        # denoise feats
        # it is empty if denoise_enabled is False
        denoise_feats = {}

        # Disable activation checkpointing for the first few recycling iters
        is_grad_enabled = torch.is_grad_enabled()
        self._disable_activation_checkpointing()

        # Main recycling loop
        num_iters = batch["aatype"].shape[-1]
        recycle_outputs = []

        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    self._enable_activation_checkpointing()
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # reuse the output for `denoise_enabled`
                if self.config.denoise_enabled and cycle_no > 0:
                    initial_rigids = outputs["sm"]["rigids"][-1]

                # denoise + training or validation, first cycle
                elif self.config.denoise_enabled and cycle_no == 0:
                    # Use `backbone_rigid_tensor_7s`` instead of `backbone_rigid_tensor`
                    # to avoid `rot-to-quat` conversion.
                    assert "backbone_rigid_tensor_7s" in feats
                    gt_aff = Rigid.from_tensor_7(feats["backbone_rigid_tensor_7s"]) # [*, N_res]
                    scaled_gt_aff = gt_aff.scale_translation(1. / self.config.structure_module.trans_scale_factor)
                    scaled_gt_trans = scaled_gt_aff.get_trans()
                    scaled_gt_rot = scaled_gt_aff.get_rots()
                    
                    # step 1: sample noise magnitude
                    if self.training:
                        noise_level = torch.randint(
                            0, self.sigmas_trans.size(0),
                            scaled_gt_trans.shape[:-2],
                            device=scaled_gt_trans.device,
                        ) # [*]
                    else:
                         noise_level = torch.randint(
                            4, 5,
                            scaled_gt_trans.shape[:-2],
                            device=scaled_gt_trans.device,
                        ) # [*]
                    used_sigmas_trans = self.sigmas_trans.to(scaled_gt_trans)[noise_level][..., None, None] # [*, 1, 1]
                    used_sigmas_rot = self.sigmas_rot.to(scaled_gt_trans)[noise_level][..., None] # [*, 1]
                    #print(self.sigmas_trans)
                    # step 2: perturb trans
                    perturbed_trans = scaled_gt_trans + torch.randn_like(scaled_gt_trans) * used_sigmas_trans # [*, N_res, 3]

                    # step 3: perturb rot
                    # step 3.1: sample rotation axis uniformly on sphere
                    rot_axis_theta = 2 * np.pi * torch.rand_like(
                        scaled_gt_trans[..., 0]
                    ) # [*, N_res]
                    rot_axis_phi = torch.arccos(2 * torch.rand_like(
                        scaled_gt_trans[..., 0]
                    ) - 1) # [*, N_res]
                    rot_axis = torch.stack([
                        torch.cos(rot_axis_theta) * torch.sin(rot_axis_phi),
                        torch.sin(rot_axis_theta) * torch.sin(rot_axis_phi),
                        torch.cos(rot_axis_phi)
                        ], dim=-1
                    ) # [*, N_res, 3]

                    # step 3.2: sample rotation angle
                    rot_angle = used_sigmas_rot * np.pi * torch.rand_like(
                        scaled_gt_trans[..., 0]
                    ) # [*, N_res] [0, pi] * scale
                    rot_angle = torch.clamp(rot_angle, min=np.pi * 1. / 180., max=175. / 180. * np.pi) # [*, N_res]
                    update_abc = torch.tan(rot_angle * 0.5)[..., None] * rot_axis
                    perturbed_rot = scaled_gt_rot.compose_q_update_vec(update_abc) # [*, N_res]

                    perturbed_rigid = Rigid(perturbed_rot, perturbed_trans)
                    initial_rigids = perturbed_rigid.to_tensor_7()
                    denoise_feats["used_sigmas_trans"] = used_sigmas_trans
                    denoise_feats["used_sigmas_rot"] = used_sigmas_rot
                # identity rigid.
                else:
                    initial_rigids = None

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev, seqs_prev = self.iteration(
                    feats,
                    m_1_prev,
                    z_prev,
                    x_prev,
                    seqs_prev,
                    initial_rigids=initial_rigids,
                    initial_seqs=None,
                    _recycle=(num_iters > 1),
                    denoise_feats=denoise_feats,
                )
                outputs.update(self.aux_heads(outputs))
                recycle_outputs.append(outputs)

        outputs = copy.copy(outputs)
        # outputs.update(self.aux_heads(outputs)) done in the loop
        outputs["recycle_outputs"] = recycle_outputs

        return outputs

    def denoise_inference_forward(
        self,
        batch,
        sigmas_trans,
        sigmas_rot,
        step_lr=1e-4,
        rot_step_lr=None,
        n_steps_each=10,
        step_schedule="squared",
        init="identity",
    ):
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        seqs_prev = None
        if rot_step_lr is None:
            rot_step_lr = step_lr
        denoise_feats = {}

        # Disable activation checkpointing for the first few recycling iters
        is_grad_enabled = torch.is_grad_enabled()
        self._disable_activation_checkpointing()

        # Main recycling loop
        num_iters = batch["aatype"].shape[-1]
        assert num_iters == 1, "denoising rigid should use `no_recycle = 0`"

        cycle_no = 0
        # Select the features for the current recycling cycle
        fetch_cur_batch = lambda t: t[..., cycle_no]
        feats = tensor_tree_map(fetch_cur_batch, batch)

        initial_rigids = None

        if step_schedule == "linear":
            total_size_trans = step_lr * sigmas_trans.sum() / sigmas_trans.max() * n_steps_each
            total_size_rot = rot_step_lr * sigmas_rot.sum() / sigmas_rot.max() * n_steps_each
        elif step_schedule == "squared":
            total_size_trans = (step_lr * (sigmas_trans / sigmas_trans.min()) ** 2).sum() / sigmas_trans.max() * n_steps_each
            total_size_rot = (rot_step_lr * (sigmas_rot / sigmas_rot.min()) ** 2).sum() / sigmas_rot.max() * n_steps_each
        else:
            raise ValueError
        print('total step size (w.r.t. one training step): translation = %g, rotation = %g' % (total_size_trans, total_size_rot))

        with torch.set_grad_enabled(False):
            step_outputs = []
            for i, (used_sigmas_trans, used_sigmas_rot) in tqdm(
                enumerate(zip(sigmas_trans, sigmas_rot)),
                total=sigmas_trans.size(0),
                desc="Denoising rigid...",
            ):
                if step_schedule == "linear":
                    step_size_trans = step_lr * used_sigmas_trans
                    step_size_rot = rot_step_lr * used_sigmas_rot
                elif step_schedule == "squared":
                    step_size_trans = step_lr * (used_sigmas_trans / sigmas_trans.min()) ** 2
                    step_size_rot = rot_step_lr * (used_sigmas_rot / sigmas_rot.min()) ** 2
                print('current step size: translation = %g, rotation = %g' % (step_size_trans, step_size_rot))

                # perform `structure_module.n_blocks` steps of update using this step_size_trans
                n_steps_this = n_steps_each
                denoise_feats = {}
                denoise_feats["used_sigmas_trans"] = used_sigmas_trans
                denoise_feats["used_sigmas_rot"] = used_sigmas_rot
                denoise_feats["denoise_step_size"] = step_size_trans
                for step in range(n_steps_this):
                    if i > 0 or step > 0:
                        initial_rigids = outputs["sm"]["rigids"][-1]
                    else:
                        if init == "identity":
                            initial_rigids = Rigid.identity(
                                feats["aatype"].shape,
                                feats["target_feat"].dtype,
                                feats["target_feat"].device,
                                self.training,
                                fmt="quat",
                            ).to_tensor_7()
                        else:
                            raise ValueError

                    outputs, m_1_prev, z_prev, x_prev, seqs_prev = self.iteration(
                        feats,
                        None,
                        None,
                        None,
                        None,
                        initial_rigids=initial_rigids,
                        initial_seqs=None,
                        _recycle=(num_iters > 1),
                        denoise_feats=denoise_feats,
                    )
                    outputs.update(self.aux_heads(outputs))
                    step_outputs.append(outputs)

        # Run auxiliary heads
        outputs = copy.copy(outputs)
        # outputs.update(self.aux_heads(outputs)) done in the loop
        outputs["recycle_step"] = step_outputs

        return outputs

