import argparse
import os
import numpy as np
import torch

from openfold.data import parsers
from openfold.np import protein, residue_constants

import multiprocessing
from functools import partial 


ROSETTA_ANTIBODY_BENCHMARK = [
    "1dlf", "1fns", "1gig", "1jfq", "1jpt", "1mfa", "1mlb", "1mqk", "1nlb", "1oaq",
    "1seq", "2adf", "2d7t", "2e27", "2fb4", "2fbj", "2r8s", "2v17", "2vxv", "2w60", 
    "2xwt", "2ypv", "3e8u", "3eo9", "3g5y", "3giz", "3gnm", "3go1", "3hc4", "3hnt",
    "3i9g", "3liz", "3lmj", "3m8o", "3mlr", "3mxw", "3nps", "3oz9", "3p0y", "3t65",
    "3umt", "3v0w", "4f57", "4h0h", "4h20", "4hpy", "4nzu",
]
THERAPEUTICS_BENCHMARK = [
    "1bey", "1cz8", "1mim", "1sy6", "1yy8", "2hwz", "3eo0", "3gkw", "3nfs", "3o2d",
    "3pp3", "3qwo", "3u0t", "4cni", "4dn3", "4g5z", "4g6k", "4hkz", "4i77", "4irz",
    "4kaq", "4m6n", "4nyl", "4od2", "4ojf", "4qxg", "4x7s", "4ypg", "5csz", "5dk3",
    "5ggq", "5ggu", "5i5k", "5jxe", "5kmv", "5l6y", "5n2k", "5nhw", "5sx4", "5tru",
    "5vh3", "5wuv", "5xxy", "5y9k", "6and",
]
LEN_ROSETTA = len(ROSETTA_ANTIBODY_BENCHMARK)
LEN_THERAPEUTICS = len(THERAPEUTICS_BENCHMARK)
assert len(np.intersect1d(ROSETTA_ANTIBODY_BENCHMARK, THERAPEUTICS_BENCHMARK)) == 0
assert len(np.unique(ROSETTA_ANTIBODY_BENCHMARK)) == LEN_ROSETTA
assert len(np.unique(THERAPEUTICS_BENCHMARK)) == LEN_THERAPEUTICS


def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]] 
        for i in range(len(aatype))
    ])
    
      
def _string_index_select(str, bool_index):
    str = np.array(list(str))
    str = list(str[bool_index])
    return ''.join(str)
    

def compute_knn(structure_1, structure_2, mask1=None, mask2=None, k=64):
    """
        Select knn residues from structure_2 against target structure_1

        Args:
            structure_1:
                [*, N1, 3] (ca) coordinate tensor of antibody
            structure_2:
                [*, N2, 3] (ca) coordinate tensor of antigen
            mask1:
                [*, N1] residue masks
            mask2:
                [*, N2] residue masks
        Returns:
            A [N2] tensor contains whether a residue is selected.
            Typically, sum([N2]) <= 64.
            
    """
    assert structure_1.ndim == 2, "only a single complex can be processed"
    if mask1 is None:
        mask1 = torch.ones_like(structure_1[..., 0])
    if mask2 is None:
        mask2 = torch.ones_like(structure_2[..., 0])
    k = min(k, int(mask2.sum().item()))
    
    d = structure_1[..., :, None, :] - structure_2[..., None, :, :] # [*, N1, N2, 3]
    
    d = d ** 2
    d = torch.sqrt(torch.sum(d, dim=-1)) # [*, N1, N2]
    d_inf = 1e8 * torch.ones_like(d)
    
    valid_pair_mask = mask1[..., :, None] * mask2[..., None, :] # [*, N1, N2]
    d = valid_pair_mask * d + (1 - valid_pair_mask) * d_inf # [*, N1, N2]
    
    d_min_value, d_min_indice = torch.min(d, dim=-2) # [*, N2]
    
    top_k_value, top_k_indice = torch.topk(d_min_value, k, dim=-1, largest=False)
    
    ret = torch.zeros_like(structure_2[..., 0]).long()
    ret[top_k_indice] = 1
    return ret


def compute_knn_np(structure_1, structure_2, mask1=None, mask2=None, k=64):
    """
        Select knn residues from structure_2 against target structure_1

        Args:
            structure_1:
                [*, N1, 3] (ca) coordinate tensor of antibody
            structure_2:
                [*, N2, 3] (ca) coordinate tensor of antigen
            mask1:
                [*, N1] residue masks
            mask2:
                [*, N2] residue masks
        Returns:
            A [N2] tensor contains whether a residue is selected.
            Typically, sum([N2]) <= 64.
            
    """
    assert structure_1.ndim == 2, "only a single complex can be processed"
    if mask1 is None:
        mask1 = np.ones_like(structure_1[..., 0])
    if mask2 is None:
        mask2 = np.ones_like(structure_2[..., 0])
    k = min(k, int(mask2.sum())) - 1
    
    d = structure_1[..., :, None, :] - structure_2[..., None, :, :] # [*, N1, N2, 3]
    
    d = d ** 2
    d = np.sqrt(np.sum(d, axis=-1)) # [*, N1, N2]
    d_inf = 1e8 * np.ones_like(d)
    
    valid_pair_mask = mask1[..., :, None] * mask2[..., None, :] # [*, N1, N2]
    d = valid_pair_mask * d + (1 - valid_pair_mask) * d_inf # [*, N1, N2]
    
    d_min_value = np.min(d, axis=-2) # [*, N2]
    
    top_k_indice = np.argpartition(d_min_value, k, axis=-1)[:k + 1]
    
    ret = np.zeros_like(structure_2[..., 0]).astype(np.int32)
    ret[top_k_indice] = 1
    return ret


def trunc_ab_ag_complex(prot: protein.Protein, k=64):
    ca_pos = residue_constants.atom_order["CA"]
    atom_positions = prot.atom_positions[..., ca_pos, :]
    atom_mask = prot.atom_mask[..., ca_pos]

    ab = atom_positions[prot.chain_index <= 1]
    ag = atom_positions[prot.chain_index > 1]
    mask_ab = atom_mask[prot.chain_index <= 1]
    mask_ag = atom_mask[prot.chain_index > 1]
    
    ag_select = compute_knn_np(ab, ag, mask_ab, mask_ag, k=k)
    filter_mask = np.zeros_like(prot.chain_index)
    filter_mask[prot.chain_index <= 1] = 1
    filter_mask[prot.chain_index > 1] = ag_select
    filter_mask = filter_mask.astype(bool)

    atom_positions_ = np.copy(prot.atom_positions)[filter_mask]
    atom_mask_ = np.copy(prot.atom_mask)[filter_mask]
    aatype_ = np.copy(prot.aatype)[filter_mask]
    residue_index_ = np.copy(prot.residue_index)[filter_mask]
    chain_index_ = np.copy(prot.chain_index)[filter_mask]
    loop_index_ = np.copy(prot.loop_index)[filter_mask]
    b_factors_ = np.copy(prot.b_factors)[filter_mask]
    
    return protein.Protein(
        atom_positions=atom_positions_,
        atom_mask=atom_mask_,
        aatype=aatype_,
        residue_index=residue_index_,
        chain_index=chain_index_,
        loop_index=loop_index_,
        b_factors=b_factors_,
    )


def filter_ab_from_complex(prot: protein.Protein):
    filter_mask = np.zeros_like(prot.chain_index)
    filter_mask[prot.chain_index <= 1] = 1
    filter_mask = filter_mask.astype(bool)

    atom_positions_ = np.copy(prot.atom_positions)[filter_mask]
    atom_mask_ = np.copy(prot.atom_mask)[filter_mask]
    aatype_ = np.copy(prot.aatype)[filter_mask]
    residue_index_ = np.copy(prot.residue_index)[filter_mask]
    chain_index_ = np.copy(prot.chain_index)[filter_mask]
    loop_index_ = np.copy(prot.loop_index)[filter_mask]
    b_factors_ = np.copy(prot.b_factors)[filter_mask]
    
    return protein.Protein(
        atom_positions=atom_positions_,
        atom_mask=atom_mask_,
        aatype=aatype_,
        residue_index=residue_index_,
        chain_index=chain_index_,
        loop_index=loop_index_,
        b_factors=b_factors_,
    )


def pdb2fasta(fname, idx, data_dir):
    basename, ext = os.path.splitext(fname)
    fpath = os.path.join(data_dir, fname)
    ret = []
    if ext == '.pdb':
        with open(fpath, 'r') as f:
            pdb_str = f.read()
        protein_object = protein.from_pdb_string_antibody(pdb_str)
        seq = _aatype_to_str_sequence(protein_object.aatype)
        
        ret.append(f">{basename}_H")
        ret.append(_string_index_select(seq, protein_object.chain_index==0))
        ret.append(f">{basename}_L")
        ret.append(_string_index_select(seq, protein_object.chain_index==1))
           
    else:
        raise ValueError(f'ext is invalid, should be either pdb of fasta, found {ext}')
    return ret, basename

def pdb2cdrfasta(fname, idx, data_dir, cdr_idx):
    basename, ext = os.path.splitext(fname)
    fpath = os.path.join(data_dir, fname)
    ret = []
    name = ['cdrh1', 'cdrh2', 'cdrh3', 'cdrl1', 'cdrl2', 'cdrl3']
    if ext == '.pdb':
        with open(fpath, 'r') as f:
            pdb_str = f.read()
        protein_object = protein.from_pdb_string_antibody(pdb_str)
        seq = _aatype_to_str_sequence(protein_object.aatype)
        
        ret.append(f">{basename}_{name[cdr_idx - 1]}")
        ret.append(_string_index_select(seq, protein_object.loop_index==cdr_idx))
           
    else:
        raise ValueError(f'ext is invalid, should be either pdb of fasta, found {ext}')
    return ret, basename

def fastas2fasta(data_dir, mode=-1):
    """merge fastas in a directory into a single fasta
    mode: format of the output fasta
        mode==-1: deepab's format: e.g.,
            >1ay1_H
            EVQLQESGPGLVKPYQSLSLSCTVTGYSITSDYAWNWIRQFPGNKLEWMGYITYSGTTDYNPSLKSRISITRDTSKNQFFLQLNSVTTEDTATYYCARYYYGYWYFDVWGQGTTLTVS
            >1ay1_L
            DIQMTQSPAIMSASPGEKVTMTCSASSSVSYMYWYQQKPGSSPRLLIYDSTNLASGVPVRFSGSGSGTSYSLTISRMEAEDAATYYCQQWSTYPLTFGAGTKLELKRA
        mode>=0: 2-line fasta, with ``mode'' unknown residues (X) inserted between heavy ang light chains.
            e.g., mode=10
            >5ggu
            QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVIWYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDPRGATLYYYYYGMDVWGQGTTVTVS\
            XXXXXXXXXX
            DIQMTQSPSSLSASVGDRVTITCRASQSINSYLDWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYYSTPFTFGPGTKVEIKRT
    """
    fnames = [x for x in os.listdir(data_dir) if x.endswith('.fasta')]
    fastas = []
    for fname in fnames:
        fpath = os.path.join(data_dir, fname)
        basename, ext = os.path.splitext(fname)
        with open(fpath, 'r') as f:
            fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(fasta_str)
        if mode >= 0:
            assert len(input_seqs) == 2
            seq = input_seqs[0] + 'X' * mode + input_seqs[1]
            fastas.append(f">{basename}_cat{mode}x")
            fastas.append(seq)
        elif mode == -1:
            for (seq, desc) in zip(input_seqs, input_descs):
                fastas.append(f">{desc}")
                fastas.append(seq)
        else:
            raise ValueError(f"unsupported mode: {mode}!")
    return fastas

        
def main(args):
    fasta = []
    job_args = os.listdir(args.data_dir)
    job_args = zip(job_args, list(range(len(job_args))))
    with multiprocessing.Pool(args.n_job) as p:
        func = partial(pdb2fasta, data_dir=args.data_dir)
        for (ret, basename) in p.starmap(func, job_args):
            assert len(ret) % 2 == 0, 'length of the returned fasta should be even'
            fasta.extend(ret)
            
    with open(args.output_path, "w") as fp:
        fp.write('\n'.join(fasta))        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=str,
        help="Path to a directory containing mmCIF or .core files"
    )
    parser.add_argument(
        "output_path", type=str,
        help="Path to output FASTA file"
    )
    parser.add_argument(
        "--n_job", type=int, default=48,
        help="number of cpu jobs"
    )
    args = parser.parse_args()

    main(args)
    
    # python scripts_narval/pdb_to_fasta.py $SCRATCH/dataset/SAbDab_database/train/ $SCRATCH/dataset/SAbDab_database/fasta_from_pdb/train_HL.fasta --n_job 12
    # python scripts_narval/pdb_to_fasta.py $SCRATCH/dataset/SAbDab_database/valid/ $SCRATCH/dataset/SAbDab_database/fasta_from_pdb/valid_HL.fasta --n_job 12    
    # python scripts_narval/pdb_to_fasta.py $SCRATCH/dataset/SAbDab_database/rosetta_antibody_benchmark/ $SCRATCH/dataset/SAbDab_database/fasta_from_pdb/rosetta_HL.fasta --n_job 12        
    # python scripts_narval/pdb_to_fasta.py $SCRATCH/dataset/SAbDab_database/therapeutics/ $SCRATCH/dataset/SAbDab_database/fasta_from_pdb/therapeutics_HL.fasta --n_job 12            