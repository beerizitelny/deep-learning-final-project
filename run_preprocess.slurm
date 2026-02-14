#!/bin/bash
#SBATCH --job-name=hd_gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate LOS_Net

./scripts/generate_HD_raw_datasets.sh /home/beerizitelny/deep-learning-final-project/big-storage/raw_data 5
