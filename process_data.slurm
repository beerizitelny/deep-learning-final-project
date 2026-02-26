#!/bin/bash
#SBATCH --job-name=hd_hot_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate LOS_Net

for DATASET in imdb imdb_test movies movies_test; do
  echo "Processing dataset: $DATASET"

  for CHUNK_ID in {1..3}; do
    echo "Running chunk $CHUNK_ID for dataset $DATASET"

    python create_HD_datasets.py \
      --dataset "$DATASET" \
      --LLM mistralai/Mistral-7B-Instruct-v0.2 \
      --base_raw_data_dir /home/beerizitelny/deep-learning-final-project/big-storage/test \
      --n_samples 10000 \
      --chunk "$CHUNK_ID"
  done

  python preprocess_datasets.py \
    --LLM mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset "$DATASET" \
    --topk_preprocess 1_000 \
    --base_raw_data_dir /home/beerizitelny/deep-learning-final-project/big-storage/raw_data \
    --base_pre_processed_data_dir /home/beerizitelny/LOS-Net/pre_processed_data \
    --input_output_type output \
    --N_max 50 \
    --input_type LOS
done
