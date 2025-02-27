#!/bin/bash
#SBATCH --job-name=train_brats_multilevel_hf_agg_res_up_simple_ref
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/waveformer-brats/results/train_brats_multilevel_hf_agg_res_up_simple_ref.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/waveformer-brats/results/train_brats_multilevel_hf_agg_res_up_simple_ref.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:4
#SBATCH --time=24:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/waveformer-brats

# Execute the Python script
srun python 3_train.py