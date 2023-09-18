import copy
from typing import Mapping, Tuple, List, Optional, Dict, Sequence

import ml_collections
import numpy as np
import torch

from openfold.data import input_pipeline


FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


def make_data_config(
    config: ml_collections.ConfigDict,
    mode: str,
    num_res: int,
) -> Tuple[ml_collections.ConfigDict, List[str]]:
    cfg = copy.deepcopy(config)
    mode_cfg = cfg[mode]
    with cfg.unlocked():
        if mode_cfg.crop_size is None:
            mode_cfg.crop_size = num_res

    feature_names = cfg.common.unsupervised_features

    if cfg[mode].supervised:
        feature_names += cfg.supervised.supervised_features

    return cfg, feature_names


def np_to_tensor_dict(
    np_example: Mapping[str, np.ndarray],
    features: Sequence[str],
) -> TensorDict:
    """Creates dict of tensors from a dict of NumPy arrays.

    Args:
        np_example: A dict of NumPy feature arrays.
        features: A list of strings of feature names to be returned in the dataset.

    Returns:
        A dictionary of features mapping feature names to features. Only the given
        features are returned, all other ones are filtered out.
    """
    tensor_dict = {
        k: torch.tensor(v) for k, v in np_example.items() if k in features
    }
    return tensor_dict


def np_example_to_features(
    np_example: FeatureDict,
    config: ml_collections.ConfigDict,
    mode: str,
):
    np_example = dict(np_example)
    num_res = int(np_example["seq_length"][0])
    cfg, feature_names = make_data_config(
        config, mode=mode, num_res=num_res
    )

    tensor_dict = np_to_tensor_dict(
        np_example=np_example,
        features=feature_names,
    )
    with torch.no_grad():
        features = input_pipeline.process_tensors_from_config(
            tensor_dict,
            cfg.common,
            cfg[mode],
        )

    return {k: v for k, v in features.items()}


class FeaturePipeline:
    def __init__(
        self,
        config: ml_collections.ConfigDict,
    ):
        self.config = config

    def process_features(
        self,
        raw_features: FeatureDict,
        mode: str = "train", 
    ) -> FeatureDict:
        return np_example_to_features(
            np_example=raw_features,
            config=self.config,
            mode=mode,
        )
