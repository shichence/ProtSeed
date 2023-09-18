import logging
logging.basicConfig(level=logging.WARNING)

import os
import copy
import pickle
from functools import partial
from typing import Optional, Sequence, List, Any

import torch
import pytorch_lightning as pl
import ml_collections as mlc

from openfold.data import data_pipeline, feature_pipeline
from openfold.utils.tensor_utils import tensor_tree_map, dict_multimap


class OpenFoldSingleDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_dir: str,
        ss_file: str,
        config: mlc.ConfigDict,
        mode: str = "train", 
        output_raw: bool = False,
    ):
        """
            Args:
                data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                config:
                    A dataset config object. See openfold.config
                mode:
                    "train", "val", or "predict"
        """
        super(OpenFoldSingleDataset, self).__init__()
        self.data_dir = data_dir
        self.ss_file = ss_file
        self.config = config
        self.mode = mode
        self.output_raw = output_raw

        valid_modes = ["train", "eval", "predict"]
        if mode not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}')

        self.chain_ids = [
            os.path.splitext(name)[0] for name in os.listdir(data_dir)
        ]    
        self.chain_id_to_idx_dict = {
            chain: i for i, chain in enumerate(self.chain_ids)
        }

        self.ss_dict = {}
        with open(self.ss_file, 'rb') as fin:
            second_structure_data = pickle.load(fin)
        logging.warning(f"get {len(second_structure_data)} second structure data")
        for ss in second_structure_data:
            self.ss_dict[ss['tag']] = ss['ss3']

        self.data_pipeline = data_pipeline.DataPipeline(self.ss_dict)
        if not self.output_raw:
            self.feature_pipeline = feature_pipeline.FeaturePipeline(config)

    def chain_id_to_idx(self, chain_id):
        return self.chain_id_to_idx_dict[chain_id]

    def idx_to_chain_id(self, idx):
        return self.chain_ids[idx]

    def __getitem__(self, idx):
        name = self.idx_to_chain_id(idx)

        if(self.mode == 'train' or self.mode == 'eval'):
            chain_id = name[4]
            path = os.path.join(self.data_dir, name)

            if os.path.exists(path + ".pdb"):
                data = self.data_pipeline.process_pdb(
                    pdb_path=path + ".pdb",
                    chain_id=chain_id,
                )
            else:
                raise ValueError("Invalid file type")

        else:
            path = os.path.join(name, name + ".fasta")
            data = self.data_pipeline.process_fasta(
                fasta_path=path,
            )

        if self.output_raw:
            return data

        feats = self.feature_pipeline.process_features(
            data, self.mode 
        )

        return feats

    def __len__(self):
        return len(self.chain_ids)


class OpenFoldDataset(torch.utils.data.Dataset):
    def __init__(self,
        datasets: Sequence[OpenFoldSingleDataset],
        probabilities: Sequence[int],
        epoch_len: int,
        generator: torch.Generator = None,
        roll_at_init: bool = True,
    ):
        self.datasets = datasets
        self.probabilities = probabilities
        self.epoch_len = epoch_len
        self.generator = generator

        def looped_shuffled_dataset_idx(dataset_len):
            while True:
                # Uniformly shuffle each dataset's indices
                weights = [1. for _ in range(dataset_len)]
                shuf = torch.multinomial(
                    torch.tensor(weights),
                    num_samples=dataset_len,
                    replacement=False,
                    generator=self.generator,
                )
                for idx in shuf:
                    yield idx

        def looped_samples(dataset_idx):
            max_cache_len = int(epoch_len * probabilities[dataset_idx])
            dataset = self.datasets[dataset_idx]
            idx_iter = looped_shuffled_dataset_idx(len(dataset))

            while True:
                cache = []
                for _ in range(max_cache_len):
                    candidate_idx = next(idx_iter)
                    cache.append(candidate_idx)

                for datapoint_idx in cache:
                    yield datapoint_idx

        self._samples = [looped_samples(i) for i in range(len(self.datasets))]

        if roll_at_init:
            self.reroll()

    def __getitem__(self, idx):
        dataset_idx, datapoint_idx = self.datapoints[idx]
        return self.datasets[dataset_idx][datapoint_idx]

    def __len__(self):
        return self.epoch_len

    def reroll(self):
        dataset_choices = torch.multinomial(
            torch.tensor(self.probabilities),
            num_samples=self.epoch_len,
            replacement=True,
            generator=self.generator,
        )

        self.datapoints = []
        for dataset_idx in dataset_choices:
            samples = self._samples[dataset_idx]
            datapoint_idx = next(samples)
            self.datapoints.append((dataset_idx, datapoint_idx))


class OpenFoldBatchCollator:
    def __init__(self, config, stage="train"):
        self.stage = stage
        self.feature_pipeline = feature_pipeline.FeaturePipeline(config)

    def __call__(self, raw_prots):
        processed_prots = []
        for prot in raw_prots:
            features = self.feature_pipeline.process_features(
                prot, self.stage
            )
            processed_prots.append(features)

        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, processed_prots) 


class OpenFoldDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, config, stage="train", generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.stage = stage    

        if generator is None:
            generator = torch.Generator()
        
        self.generator = generator
        self._prep_batch_properties_probs()

    def _prep_batch_properties_probs(self):
        keyed_probs = []
        stage_cfg = self.config[self.stage]

        max_iters = self.config.common.max_recycling_iters
        if stage_cfg.supervised:
            # Supplement 1.11.5
            # In 90% of training mini-batches the FAPE backbone loss is clamped by e_max = 10A.
            # In the remaining 10% it is not clamped, e_max = + \infty.
            # For side-chains it is always clamped by e_max = 10A.
            clamp_prob = self.config.supervised.clamp_prob
            keyed_probs.append(
                ("use_clamped_fape", [1 - clamp_prob, clamp_prob])
            )
        
        if stage_cfg.uniform_recycling:
            recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
        else:
            recycling_probs = [
                0. for _ in range(max_iters + 1)
            ]
            recycling_probs[-1] = 1.
        
        keyed_probs.append(
            ("no_recycling_iters", recycling_probs)
        )

        keys, probs = zip(*keyed_probs)
        max_len = max([len(p) for p in probs])
        padding = [[0.] * (max_len - len(p)) for p in probs] 
        
        self.prop_keys = keys
        self.prop_probs_tensor = torch.tensor(
            [p + pad for p, pad in zip(probs, padding)],
            dtype=torch.float32,
        )

    def _add_batch_properties(self, batch):
        samples = torch.multinomial(
            self.prop_probs_tensor,
            num_samples=1, # 1 per row
            replacement=True,
            generator=self.generator
        )

        aatype = batch["aatype"]
        batch_dims = aatype.shape[:-2]
        recycling_dim = aatype.shape[-1]
        no_recycling = recycling_dim
        for i, key in enumerate(self.prop_keys):
            sample = int(samples[i][0])
            sample_tensor = torch.tensor(
                sample, 
                device=aatype.device, 
                requires_grad=False
            )
            orig_shape = sample_tensor.shape
            sample_tensor = sample_tensor.view(
                (1,) * len(batch_dims) + sample_tensor.shape + (1,)
            )
            sample_tensor = sample_tensor.expand(
                batch_dims + orig_shape + (recycling_dim,)
            )
            batch[key] = sample_tensor

            if key == "no_recycling_iters":
                no_recycling = sample 
        
        resample_recycling = lambda t: t[..., :no_recycling + 1]
        batch = tensor_tree_map(resample_recycling, batch)

        return batch

    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._add_batch_properties(batch)

        return _batch_prop_gen(it)


class OpenFoldDataModule(pl.LightningDataModule):
    def __init__(self,
        config: mlc.ConfigDict,
        train_data_dir: Optional[str] = None,
        val_data_dir: Optional[str] = None,
        predict_data_dir: Optional[str] = None,
        ss_file: Optional[str] = None,
        batch_seed: Optional[int] = None,
        train_epoch_len: Optional[int] = None,
        **kwargs
    ):
        super(OpenFoldDataModule, self).__init__()

        self.config = config
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.predict_data_dir = predict_data_dir
        self.ss_file = ss_file
        self.batch_seed = batch_seed
        self.train_epoch_len = train_epoch_len

        if self.train_data_dir is None and self.predict_data_dir is None:
            raise ValueError(
                'At least one of train_data_dir or predict_data_dir must be '
                'specified'
            )

        if self.ss_file is None:
            raise ValueError(
                'secondary structure data should be provided.'
            )
        self.training_mode = self.train_data_dir is not None


    def setup(self):
        if self.training_mode:
            train_dataset = OpenFoldSingleDataset(
                data_dir=self.train_data_dir,
                ss_file=self.ss_file,
                config=self.config,
                mode="train",
                output_raw=True,
            )

            datasets = [train_dataset]
            probabilities = [1.]   
            train_epoch_len = self.train_epoch_len or \
                              sum([len(_) for _ in datasets])

            self.train_dataset = OpenFoldDataset(
                datasets=datasets,
                probabilities=probabilities,
                epoch_len=train_epoch_len,
                generator=None,
                roll_at_init=False,
            )

            if self.val_data_dir is not None:
                self.eval_dataset = OpenFoldSingleDataset(
                    data_dir=self.val_data_dir,
                    ss_file=self.ss_file,
                    config=self.config,
                    mode="eval",
                    output_raw=True,
                )
            else:
                self.eval_dataset = None

        else:           
            self.predict_dataset = OpenFoldSingleDataset(
                data_dir=self.predict_data_dir,
                ss_file=self.ss_file,
                config=self.config,
                mode="predict",
                output_raw=False,
            )

    def _gen_dataloader(self, stage):
        generator = torch.Generator()
        if self.batch_seed is not None:
            generator = generator.manual_seed(self.batch_seed)

        dataset = None
        if stage == "train":
            dataset = self.train_dataset
            dataset.reroll()
        elif stage == "eval":
            dataset = self.eval_dataset
        elif stage == "predict":
            dataset = self.predict_dataset
        else:
            raise ValueError("Invalid stage")

        batch_collator = OpenFoldBatchCollator(self.config, stage)

        dl = OpenFoldDataLoader(
            dataset,
            config=self.config,
            stage=stage,
            generator=generator,
            batch_size=self.config.data_module.data_loaders.batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            pin_memory=self.config.data_module.data_loaders.pin_memory,
            collate_fn=batch_collator,
        )

        return dl

    def train_dataloader(self):
        return self._gen_dataloader("train") 

    def val_dataloader(self):
        if self.eval_dataset is not None:
            return self._gen_dataloader("eval")
        return None

    def predict_dataloader(self):
        return self._gen_dataloader("predict") 


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, batch_path):
        with open(batch_path, "rb") as f:
            self.batch = pickle.load(f)

    def __getitem__(self, idx):
        return copy.deepcopy(self.batch)

    def __len__(self):
        return 1000


class DummyDataLoader(pl.LightningDataModule):
    def __init__(self, batch_path):
        super().__init__()
        self.dataset = DummyDataset(batch_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset)
