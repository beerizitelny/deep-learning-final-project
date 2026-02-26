# LOS-Net

This repository contains the official code of the paper
**Beyond Next Token Probabilities: Learnable, Fast Detection of Hallucinations and Data Contamination on LLM Output Distributions** [https://arxiv.org/pdf/2503.14043]

<p align="center">
  <img src="./Figures/LOS.png" width="100%" height="50%">
</p>

# Installation

First create a conda environment
```
conda env create -f los_net_env.yml
```
and activate it
```
conda activate los_net_env
```

# Handeling Datasets
## HD
### Generating Raw Datasets (HD)
#### Available Datasets & Supported Models
Supported datasets are:
- `imdb`,`imdb_test` 
- `movies`, `movies_test`

Supported models for all datasets:
- `mistralai/Mistral-7B-Instruct-v0.2`

#### Dataset Construction Instructions:
To construct a dataset for a specific model, execute the following command, replacing placeholders with the desired dataset and model:

```bash
python create_HD_datasets.py \
  --dataset <DATASET_NAME> \
  --LLM <MODEL_NAME> \
  --base_raw_data_dir <BASE_RAW_DATA_DIRECTORY> \
  --n_samples <NUMBER_OF_SAMPLES_TO_USE_FROM_DATASET (default is 10_000 which effectivly is everything)> \
  --chunk <CHUNK_ID> 
```
**Note:** We split the generation to 10 chunks (for efficiency reasons), indexed from 1 to 10, where the i-th chunk stand for samples i000 -> (i+1)000.

- **Example:** 
  ```bash
  python create_HD_datasets.py \
    --dataset imdb \
    --LLM mistralai/Mistral-7B-Instruct-v0.2 \
    --base_raw_data_dir /home/guy_b/big-storage/raw_data \
    --n_samples 10_000 \
    --chunk 1
  ```

To automatically generate all the raw (full) datasets for all the models, datasets and chunks, run the following command:

**use given batch file**
   ```bash
   sbatch process_data.slurm
   ```


## Preprocess Raw Datasets

To preprocess a raw dataset for a specific model (to create the proper datatype for LOS-Net), use the following commands:
(the given batch file process_data.slurm already does it)

```bash
python preprocess_datasets.py \
  --LLM <MODEL_NAME> \
  --dataset <DATASET_NAME> \
  --base_raw_data_dir <BASE_RAW_DATA_DIRECTORY> \
  --topk_preprocess <TOP_K> \
  --base_pre_processed_data_dir <BASE_PRE_PROCESSED_DATA_DIRECTORY> \
  --input_output_type <input/output> \
  --N_max <MAX_SEQUENCE_LENGTH> \
  --input_type LOS
```
- **Example:** 
  ```bash
  python preprocess_datasets.py \
    --LLM EleutherAI/pythia-6.9b \
    --dataset WikiMIA_32 \
    --topk_preprocess 1_000_000 \
    --base_raw_data_dir /home/guy_b/big-storage/raw_data \
    --base_pre_processed_data_dir /home/guy_b/LOS-Net/pre_processed_data \
    --input_output_type input \
    --N_max 100 \
    --input_type LOS
  ```


# Reproducibility
## Standard Experiemnts 
To reproduce the experiments from the paper, run the commands below:


- HD:
  | Dataset  | Command |
  |------------|---------|
  | **mistral -- IMDB** | ```wandb sweep ./sweeps/LOS/DC/mistral_imdb_output.yaml``` | 
  | **mistral -- Movies** | ```wandb sweep ./sweeps/LOS/DC/mistral_movies_output.yaml``` |

In order to test all approaches you need to edit the running files:
1. **DOLA**
under "create_HD_dataset.py" update USE_DOLA = True and run the batch file.
make sure to use different base_dir so you won't override code-base generated-data.
After DOLA data is generated, you can run the experiment as written. 
2. **gradient clipping**
under "main.py" update GRAD_CLIP = 1.0 .
if the (regular) data is already generated, you can run the experiment as written.