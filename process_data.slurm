#!/bin/bash
#SBATCH --job-name=hd_hot_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate LOS_Net

# ./scripts/generate_HD_raw_datasets.sh /home/beerizitelny/deep-learning-final-project/big-storage/raw_data 5

# python create_HD_datasets.py \
#  --dataset imdb \
#  --LLM mistralai/Mistral-7B-Instruct-v0.2 \
#  --base_raw_data_dir /home/beerizitelny/deep-learning-final-project/big-storage/raw_data \
#  --n_samples 10_000 \
#  --chunk 1

for CHUNK_ID in {1..10}; do
  echo "Running chunk $CHUNK_ID"

  python create_HD_datasets.py \
    --dataset imdb \
    --LLM mistralai/Mistral-7B-Instruct-v0.2 \
    --base_raw_data_dir /home/beerizitelny/deep-learning-final-project/big-storage/raw_data \
    --n_samples 10000 \
    --chunk "$CHUNK_ID"
done
