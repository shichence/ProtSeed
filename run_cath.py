import argparse
import logging
logging.basicConfig(level=logging.INFO)

import os
import sys
import time
import pickle
import numpy as np

import torch

from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.feats import atom14_to_atom37
from openfold.utils.seed import seed_everything
from openfold.utils.superimposition import superimpose
from openfold.np import residue_constants, protein
from openfold.model.model import AlphaFold
from openfold.data import feature_pipeline, data_pipeline, data_transforms
from openfold.config import model_config

import debugger


def main(args):
    if args.seed is not None:
        seed_everything(args.seed)

    config = model_config(
        name=args.config_preset,
        yaml_config_preset=args.yaml_config_preset,
        train=False,
        low_prec=False,
    )
    model = AlphaFold(config)
    model = model.eval()

    # Load the checkpoint
    ## deepspeed
    if args.deepspeed:
        latest_path = os.path.join(args.resume_from_ckpt, 'latest')
        if os.path.isfile(latest_path):
            with open(latest_path, 'r') as fd:
                tag_ = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")
        ckpt_path = os.path.join(args.resume_from_ckpt,
                                tag_, "mp_rank_00_model_states.pt")
        ckpt_epoch = os.path.basename(args.resume_from_ckpt).split('-')[0]
        if args.ema:
            state_dict = torch.load(ckpt_path, map_location="cpu")["ema"]["params"]
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")["module"]
            state_dict = {k[len("module.model."):]: v for k,
                        v in state_dict.items()}
    ## native fp32 or mixed fp16
    else:
        ckpt_path = args.resume_from_ckpt
        ckpt_epoch = os.path.basename(args.resume_from_ckpt).split('-')[0]
        if args.ema:
            state_dict = torch.load(ckpt_path, map_location="cpu")["ema"]["params"]
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            state_dict = {k[len("model."):]: v for k,
                        v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.model_device)
    logging.info(f"Successfully loaded model weights from {ckpt_path}...")

    # Prepare data
    ss_dict = {}
    with open(args.ss_file, 'rb') as fin:
        second_structure_data = pickle.load(fin)
    logging.warning(f"get {len(second_structure_data)} second structure data")
    for ss in second_structure_data:
        ss_dict[ss['tag']] = ss['ss3']

    data_processor = data_pipeline.DataPipeline(ss_dict)
    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get input 
    fname = os.path.basename(args.pdb_path)
    name = os.path.splitext(fname)[0]
    chain_id = name[4]

    feature_dict = data_processor.process_pdb(
        pdb_path=args.pdb_path,
        chain_id=chain_id,
    )
    feature_dict["no_recycling_iters"] = args.no_recycling_iters

    processed_feature_dict = feature_processor.process_features(
        feature_dict,
        mode="predict",
    )

    logging.info("Executing model...")
    batch = processed_feature_dict
    with torch.no_grad():
        batch = {
            k: torch.as_tensor(v, device=args.model_device)
            for k, v in batch.items()
        }
        t = time.perf_counter()

        # add batch dim
        batch = tensor_tree_map(lambda x: x[None], batch)
        if not config.model.denoise_enabled:
            out = model(batch)
        else:
            #out = model(batch)
            out = model.denoise_inference_forward(
                batch=batch, 
                sigmas_trans=model.sigmas_trans.data.clone(),
                sigmas_rot=model.sigmas_rot.data.clone(),
                step_lr=1e-5,
                n_steps_each=10,
                step_schedule="squared",
            )

        # rmsd
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords = batch["all_atom_positions"][..., -1].float() # [*, N, 37, 3]
        pred_coords = out["final_atom_positions"].float() # [*, N, 37, 3]
        all_atom_mask = batch["all_atom_mask"][..., -1].float() # [*, N, 37]
        gt_coords_masked = gt_coords * all_atom_mask[..., None] # [*, N, 37, 3]
        pred_coords_masked = pred_coords * all_atom_mask[..., None] # [*, N, 37, 3]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :] # [*, N, 3]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :] # [*, N, 3]
        superimposed_pred, rmsd = superimpose(
            gt_coords_masked_ca, pred_coords_masked_ca
        ) # [*, N, 3]

        # remove batch dim
        batch = tensor_tree_map(lambda x: x[0], batch)
        # the following operation is incorrect
        # the shape of out["sm"]["positions"]
        # is [8, 1, N, 14, 3], instead of [1, 8, N, 14, 3]
        # out = tensor_tree_map(lambda x: x[0], out)

        logging.info(f"Inference time: {time.perf_counter() - t}")

    # Toss out the recycling dimensions --- we don't need them anymore
    batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)

    # handle the discrepancy caused by predicted aatype.
    if "final_aatype" in out:
        fake_batch = {"aatype": out["final_aatype"]}
        fake_batch = data_transforms.make_atom14_masks(fake_batch)
        out["final_atom_positions"] = atom14_to_atom37(
            out["sm"]["positions"][-1], fake_batch
        )
        out["residx_atom14_to_atom37"] = fake_batch["residx_atom14_to_atom37"]
        out["residx_atom37_to_atom14"] = fake_batch["residx_atom37_to_atom14"]
        out["atom14_atom_exists"] = fake_batch["atom14_atom_exists"]
        out["atom37_atom_exists"] = fake_batch["atom37_atom_exists"]
        out["final_atom_mask"] = fake_batch["atom37_atom_exists"]
    # convert torch data to numpy data.
    # don't forget to toss out the batch dimension
    tmp_out = tensor_tree_map(lambda x: np.array(x[0].cpu()), out)

    plddt = tmp_out["plddt"] # [N]
    mean_plddt = np.mean(plddt) # [,]
    # [N, 37]
    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    # aar
    gt_aatype = batch["aatype"]
    final_pred_aatype = out["final_aatype"].cpu().numpy()[0]
    aar = (final_pred_aatype == gt_aatype).sum() / len(gt_aatype)
    rmsd = rmsd.item()

    # ppl
    final_seqs_distribution = out["final_seqs"].cpu().numpy()[0] #[N, 21]
    negative_logp = 0
    for i in range(len(gt_aatype)):
        negative_logp += (-np.log(final_seqs_distribution[i, gt_aatype[i]]))
    negative_logp = negative_logp / len(gt_aatype)
    ppl = np.exp(negative_logp)
    print(f"{name}, aar: {aar}, rmsd: {rmsd}, ppl: {ppl}")

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=tmp_out,
        b_factors=plddt_b_factors,
        chain_index=batch["chain_index"],
    )

    # Save the unrelaxed PDB.
    unrelaxed_output_path = os.path.join(
        args.output_dir,
        f"{name}_{ckpt_epoch}_{args.config_preset}_rec{args.no_recycling_iters}_s{args.seed}_unrelaxed.pdb"
    )
    with open(unrelaxed_output_path, 'w') as f:
        f.write(protein.to_pdb(unrelaxed_protein))

    if args.relax:
        if "relax" not in sys.modules:
            import openfold.np.relax.relax as relax

        logging.info("start relaxation")
        amber_relaxer = relax.AmberRelaxation(
            use_gpu=(args.model_device != "cpu"),
            **config.relax,
        )
        try:
            # Relax the prediction.
            t = time.perf_counter()
            visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
            if("cuda" in args.model_device):
                device_no = args.model_device.split(":")[-1]
                os.environ["CUDA_VISIBLE_DEVICES"] = device_no
            relaxed_pdb_str, _, _ = amber_relaxer.process(
                prot=unrelaxed_protein)
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
            logging.info(f"Relaxation time: {time.perf_counter() - t}")

            # Save the relaxed PDB.
            relaxed_output_path = os.path.join(
                args.output_dir,
                f"{name}_{ckpt_epoch}_{args.config_preset}_rec{args.no_recycling_iters}_s{args.seed}_relaxed.pdb"
            )
            with open(relaxed_output_path, 'w') as f:
                f.write(relaxed_pdb_str)
        except Exception as e:
            logging.warning(e)
            logging.warning("relaxation failed...")


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pdb_path", type=str,
    )
    parser.add_argument(
        "ss_file", type=str,
    )
    parser.add_argument(
        "resume_from_ckpt", type=str,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--deepspeed", type=bool_type, default=False,
        help="Checkpoint type"
    )
    parser.add_argument(
        "--relax", type=bool_type, default=True,
        help="Whether to perform the relaxation"
    )
    parser.add_argument(
        "--ema", type=bool_type, default=True,
        help="Whether to use ema model parameters"
    )
    parser.add_argument(
        "--no_recycling_iters", type=int, default=3,
        help="number of recycling iterations"
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )
    parser.add_argument(
        "--model_device", type=str, default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--config_preset", type=str, default=None,
        help=(
            "Config setting. Choose e.g. 'initial_training', 'finetuning', "
            "'model_1', etc. By default, the actual values in the config are "
            "used."
        )
    )
    parser.add_argument(
        "--yaml_config_preset", type=str, default=None,
        help=(
            "A path to a yaml file that contains the updated config setting. "
            "If it is set, the config_preset will be overwrriten as the basename "
            "of the yaml_config_preset."
        )
    )
    parser.add_argument(
        "--cpus", type=int, default=10,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    if args.model_device == "cpu" and torch.cuda.is_available():
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    if args.config_preset is None and args.yaml_config_preset is None:
        raise ValueError(
            "Either --config_preset or --yaml_config_preset should be specified."
        )

    if args.yaml_config_preset is not None:
        if not os.path.exists(args.yaml_config_preset):
            raise FileNotFoundError(
                f"{os.path.abspath(args.yaml_config_preset)}")
        args.config_preset = os.path.splitext(
            os.path.basename(args.yaml_config_preset)
        )[0]
        logging.info(
            f"the config_preset is set as {args.config_preset} by yaml_config_preset.")

    main(args)
