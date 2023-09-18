import logging
logging.basicConfig(level=logging.WARNING)

import os
import numpy as np
from typing import Mapping, Optional, Sequence, Any

from openfold.data import parsers
from openfold.np import residue_constants, protein

FeatureDict = Mapping[str, np.ndarray]


def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]] 
        for i in range(len(aatype))
    ])


def make_sequence_features(
    sequence: str,
    ss: str,
    description: str,
    num_res: int,
    chain_index: Optional[np.ndarray] = None,
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["sstype"] = residue_constants.ss_to_onehot(
        ss=ss,
        mapping=residue_constants.second_structures_order,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=np.object_
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    if chain_index is not None:
        chain_index = chain_index.astype(np.int32)
        features["chain_index"] = chain_index
        chain_gap = 100 * np.ones(num_res, dtype=np.int32)
        features["residue_index"] += (chain_index * chain_gap)
            
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=np.object_
    )
    return features


def make_protein_features(
    protein_object: protein.Protein,
    ss: str,
    description: str,
    normalize_coordinates: bool = False,
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    chain_index = protein_object.chain_index
    
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            ss=ss,
            description=description,
            num_res=len(protein_object.aatype),
            chain_index=chain_index,
        )
    )

    all_atom_positions = protein_object.atom_positions # [num_res, num_atom_type, 3]
    all_atom_mask = protein_object.atom_mask # [num_res, num_atom_type]

    if normalize_coordinates:
        xyz_mean = np.sum(all_atom_positions, (0, 1), keepdims=True) / np.sum(all_atom_mask)
        all_atom_positions = all_atom_positions - xyz_mean

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)

    return pdb_feats


def make_pdb_features(
    protein_object: protein.Protein,
    ss: str,
    description: str,
    normalize_coordinates: bool = True,
) -> FeatureDict:
    
    pdb_feats = make_protein_features(
        protein_object,
        ss,
        description,
        normalize_coordinates=normalize_coordinates,
    )
    return pdb_feats


class DataPipeline:
    """Assembles input features."""
    def __init__(self, ss_dict):
        self.ss_dict = ss_dict

    def process_fasta(
        self,
        fasta_path: str,
    ) -> FeatureDict:
        """Assembles features for a single sequence in a FASTA file""" 
        with open(fasta_path, 'r') as f:
            fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(fasta_str)

        if len(input_seqs) != 1:
            raise ValueError(
                "Parsing fasta for general proteins...\n"
                f"More than one input sequence found in {fasta_path}."
            )
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        ss = self.ss_dict[input_description]
        chain_index = np.array([0] * len(input_seqs[0]))

        num_res = len(input_sequence)

        sequence_features = make_sequence_features(
            sequence=input_sequence,
            ss=ss,
            description=input_description,
            num_res=num_res,
            chain_index=chain_index,
        )

        return {
            **sequence_features,
        }

    def process_pdb(
        self,
        pdb_path: str,
        chain_id: Optional[str] = None,
    ) -> FeatureDict:
        """
            Assembles features for a protein in a PDB file.
        """
        tag = os.path.splitext(
            os.path.basename(pdb_path)
        )[0]
        with open(pdb_path, 'r') as f:
            pdb_str = f.read()

        protein_object = protein.from_pdb_string(pdb_str, chain_id)
        ss = self.ss_dict[tag]
        description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
        # logging.info(f"cur desc: {description}")
        pdb_feats = make_pdb_features(
            protein_object,
            ss,
            description,
            normalize_coordinates=True,
        )

        return {
            **pdb_feats,
        }
