# Protseed-Cath

## Environment
We provide a conda environment file `scripts/environment.yml` for Linux machine with CUDA 11.3 / 11.4 installed. Pleaes modify the `environment.yml` if you want to use other CUDA version. To install the environment, run
```bash
bash scripts/install_third_party_dependencies.sh
```
Replace the `mamba` with `conda` in the script if you do not use `mambaforge`.

## Checkpoints and Data
We provide the checkpoints and data used in the paper in [Google Drive](https://drive.google.com/drive/folders/1nns6Js4vIcS3QSHRJG8BaHAJlGy_iXdF?usp=sharing). Please download the checkpoints and put them in the `checkpoints` folder. The md5sum of the checkpoints and data are listed in `scripts/md5sum.txt`.

If you want to train your own model, please modify the data path in `scripts/train.sh`.

## Evaluation
Please see `scripts/eval.sh` for the evaluation script. 


```bash
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
```

## Training
Please see `scripts/train.sh` for the training script.

```bash
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
```

Note that we use `wandb` to record the training process. Please modify the `wandb_entity` and `wandb_project` if you want to use your own `wandb` account.

# Protseed-Antibody Generation

*Coming very soon!*