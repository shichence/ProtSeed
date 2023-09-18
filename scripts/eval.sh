#! /bin/bash
# 1. Activate environement
ENV_NAME=protseed
source activate $ENV_NAME
echo env done

TRAIN_DIR=$SCRATCH/datasets/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=$SCRATCH/datasets/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
TEST_DIR=$SCRATCH/datasets/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_test
ckpt_path="checkpoints/epoch99-step49999-val_loss=3.581.ckpt"

python run_cath.py \
    $TEST_DIR/3tm4A01.pdb \
    $SCRATCH/datasets/structure_datasets/cath/raw/ss_annotation_31885.pkl \
    $ckpt_path \
    --yaml_config_preset yaml_config/deterministic.yml \
    --output_dir $SCRATCH/projects_output/cath_gen/inference/eval \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --deepspeed false \
    --relax false \
    --seed 42

python batch_run_cath.py \
    $TEST_DIR \
    $SCRATCH/datasets/structure_datasets/cath/raw/ss_annotation_31885.pkl \
    $ckpt_path \
    --yaml_config_preset yaml_config/deterministic.yml \
    --output_dir $SCRATCH/projects_output/cath_gen/inference/eval/deterministic \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \
    --deepspeed false \
    --seed 9141423
