#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate LOS_Net

chmod +x ./scripts/preprocess_raw_datasets_LOS.sh

bash ./scripts/preprocess_raw_datasets_LOS.sh /home/beerizitelny/deep-learning-final-project/big-storage/raw_data /home/beerizitelny/LOS-Net/pre_processed_data 5