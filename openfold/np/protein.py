"""Protein data type."""
import dataclasses
import io
from typing import Any, Mapping, Optional
import re

from openfold.np import residue_constants
from Bio.PDB import PDBParser
import numpy as np

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.
PICO_TO_ANGSTROM = 0.01

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f'Cannot build an instance with more than {PDB_MAX_CHAINS} chains '
                'because these cannot be written to PDB format.')


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Args:
        pdb_str: The contents of the pdb file
        chain_id: If None, then the pdb file must contain a single chain (which
        will be parsed). If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
        A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != ' ':
                raise ValueError(
                        f'PDB contains an insertion code at chain {chain.id} and residue '
                        f'index {res.id[1]}. These are not supported.')
            res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
            restype_idx = residue_constants.restype_order.get(
                    res_shortname, residue_constants.restype_num)
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    chain_ids = np.array(chain_ids)
    _, idx = np.unique(chain_ids, return_index=True)
    unique_chain_ids = chain_ids[np.sort(idx)]
    
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
    )


def from_pdb_string_antibody(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """A variant of func::from_pdb_string for antibody pdb data.
    
    WARNING: The insertion code is explicitly handled in this function.  
    
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f'Only single model PDBs are supported. Found {len(models)} models.')
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    # define constants of 6 cdr loops
    # now do not add 2 anchor nodes at each end
    # https://www.researchgate.net/figure/CDR-definitions-in-Chothia-numbering_tbl1_337735681
    loop_index = []
    CDR_H1_RANGE_WITH_ANCHOR = (26, 32) # 1
    CDR_H2_RANGE_WITH_ANCHOR = (52, 56) # 2
    CDR_H3_RANGE_WITH_ANCHOR = (95, 102) # 3
    CDR_L1_RANGE_WITH_ANCHOR = (24, 34) # 4
    CDR_L2_RANGE_WITH_ANCHOR = (50, 56) # 5
    CDR_L3_RANGE_WITH_ANCHOR = (89, 97) # 6
    def is_in_range(x, range_slice):
        return x >= range_slice[0] and x <= range_slice[1]
    
    # a workaround for insertion code
    for chain in model:
        insertion_code_offset = 0

        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            loop_index_ = 0
            if chain.id == 'H':
                if is_in_range(res.id[1], CDR_H1_RANGE_WITH_ANCHOR):
                    loop_index_ = 1
                elif is_in_range(res.id[1], CDR_H2_RANGE_WITH_ANCHOR):
                    loop_index_ = 2
                elif is_in_range(res.id[1], CDR_H3_RANGE_WITH_ANCHOR):
                    loop_index_ = 3

            elif chain.id == 'L':
                if is_in_range(res.id[1], CDR_L1_RANGE_WITH_ANCHOR):
                    loop_index_ = 4
                elif is_in_range(res.id[1], CDR_L2_RANGE_WITH_ANCHOR):
                    loop_index_ = 5
                elif is_in_range(res.id[1], CDR_L3_RANGE_WITH_ANCHOR):
                    loop_index_ = 6
            
            if res.id[2] != ' ':
                insertion_code_offset += 1
            res_shortname = residue_constants.restype_3to1.get(
                res.resname, 'X')
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num)
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            if loop_index_ == 0:
                aatype.append(restype_idx)
            else:
                aatype.append(restype_idx)
                # aatype.append(residue_constants.restype_num)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1] + insertion_code_offset)
            chain_ids.append(chain.id)
            loop_index.append(loop_index_)
            b_factors.append(res_b_factors)


    # Chain IDs are usually characters so map these to ints.
    # We want H:0, L:1, other chains 2,3,4,...
    # np.unique with order preserving
    chain_ids = np.array(chain_ids)
    _, idx = np.unique(chain_ids, return_index=True)
    unique_chain_ids = chain_ids[np.sort(idx)]
    
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
    )


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = 'TER'
    return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
            f'{chain_name:>1}{residue_index:>4}')


def to_pdb(prot: Protein, output_mask: Optional[np.ndarray] = None) -> str:
    """Converts a `Protein` instance to a PDB string.
    Args:
            prot: The protein to convert to PDB.
            output_mask: The residual level mask indicating whether to keep the residual or not.
    Returns:
            PDB string.
    """
    restypes = residue_constants.restypes + ['X']
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    if output_mask is not None:
        atom_mask = atom_mask * output_mask[..., None]
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError('Invalid aatypes.')

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append('MODEL     1')
    atom_index = 1
    last_chain_index = chain_index[0]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            last_chain_index = chain_index[i]
            # Ignore the X residue
            if res_1to3(aatype[i - 1]) != "UNK":
                pdb_lines.append(_chain_end(
                    atom_index, res_1to3(
                        aatype[i - 1]), chain_ids[chain_index[i - 1]],
                    residue_index[i - 1]))
                atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
                atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            record_type = 'ATOM'
            name = atom_name if len(atom_name) == 4 else f' {atom_name}'
            alt_loc = ''
            insertion_code = ''
            occupancy = 1.00
            # Protein supports only C, N, O, S, this works.
            element = atom_name[0]
            charge = ''
            # PDB is a columnar format, every space matters here!
            atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                         f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                         f'{residue_index[i]:>4}{insertion_code:>1}   '
                         f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                         f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                         f'{element:>2}{charge:>2}')
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]),
                                chain_ids[chain_index[-1]], residue_index[-1]))
    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'  # Add terminating newline. 


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    chain_index: Optional[np.ndarray] = None,
) -> Protein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.

    Returns:
      A protein instance.
    """
    if b_factors is None:
        b_factors = np.zeros_like(result["final_atom_mask"])
    if chain_index is None:
        chain_index = np.zeros_like(features["aatype"])
    if "final_aatype" in result:
        aatype = result["final_aatype"]
    else:
        aatype = features["aatype"]

    return Protein(
        aatype=aatype,
        atom_positions=result["final_atom_positions"],
        atom_mask=result["final_atom_mask"],
        residue_index=features["residue_index"] + 1,
        chain_index=chain_index,
        b_factors=b_factors,
    )
