import torch
import torch.nn as nn

from openfold.model.primitives import Linear, LayerNorm
from openfold.utils.loss import (
    compute_plddt,
    compute_tm,
    compute_predicted_aligned_error,
)


class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()

        key2head_fn = {
            "lddt": PerResidueLDDTCaPredictor,
            "distogram": DistogramHead,
            "latent_distogram": DistogramHead,
            "experimentally_resolved": ExperimentallyResolvedHead,
        }
        for key in key2head_fn:
            if config[key]["weight"] > 0:
                attr_name = key if key != "lddt" else "plddt"
                setattr(self, attr_name, key2head_fn[key](**config[key]))

        if config.tm.enabled:
            self.tm = TMScoreHead(
                **config.tm,
            )

        self.config = config

    def forward(self, outputs):
        aux_out = {}
        
        if hasattr(self, "plddt"):
            # plddt for last step's single representation
            lddt_logits = self.plddt(outputs["sm"]["single"])
            aux_out["lddt_logits"] = lddt_logits
            aux_out["plddt"] = compute_plddt(lddt_logits)

            # plddt for each step's single representation
            # lddt_logits = self.plddt(outputs["sm"]["singles"])
            # aux_out["lddt_logits_by_sm_step"] = lddt_logits
            # aux_out["plddt_by_sm_step"] = compute_plddt(lddt_logits)

        if hasattr(self, "distogram"):
            # outputs of evoformer
            distogram_logits = self.distogram(outputs["pair"])
            aux_out["distogram_logits"] = distogram_logits


        if hasattr(self, "latent_distogram"):
            # outputs of evoformer
            latent_distogram_logits = self.latent_distogram(outputs["latent_pair"])
            aux_out["latent_distogram_logits"] = latent_distogram_logits


        if hasattr(self, "experimentally_resolved"):
            # outputs of evoformer
            experimentally_resolved_logits = self.experimentally_resolved(
                outputs["single"]
            )
            aux_out[
                "experimentally_resolved_logits"
            ] = experimentally_resolved_logits

        if self.config.tm.enabled:
            tm_logits = self.tm(outputs["pair"])
            aux_out["tm_logits"] = tm_logits
            aux_out["predicted_tm_score"] = compute_tm(
                tm_logits, **self.config.tm
            )
            aux_out.update(
                compute_predicted_aligned_error(
                    tm_logits,
                    **self.config.tm,
                )
            )

        return aux_out


class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, no_bins, c_in, c_hidden, **kwargs):
        super(PerResidueLDDTCaPredictor, self).__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_3 = Linear(self.c_hidden, self.no_bins, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of bins
        """
        super(TMScoreHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pairwise embedding
        Returns:
            [*, N_res, N_res, no_bins] prediction
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        return logits


class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    """

    def __init__(self, c_s, c_out, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super(ExperimentallyResolvedHead, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        # [*, N, C_out]
        logits = self.linear(s)
        return logits
