#! /bin/bash

ENV_NAME=protseed
source activate $ENV_NAME
echo env done

TRAIN_DIR=$SCRATCH/datasets/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=$SCRATCH/datasets/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
TEST_DIR=$SCRATCH/datasets/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_test
OUTPUT_DIR=$SCRATCH/projects_output/cath_gen/release_output

python train_cath.py $TRAIN_DIR $OUTPUT_DIR \
    --ss_file $SCRATCH/datasets/structure_datasets/cath/raw/ss_annotation_31885.pkl \
    --val_data_dir $VALID_DIR \
    --seed 2022 \
    --yaml_config_preset yaml_config/deterministic.yml \
    --precision 16 --gpus 4 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version release \
    --wandb_project cath_gen \
    --train_epoch_len 2000 \
    --gradient_clip_val 1.0