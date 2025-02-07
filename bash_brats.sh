#!/bin/bash
#SBATCH --job-name=train_brats_ablation_pooling
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/waveformer-brats/results/brats_ablation_pooling.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/waveformer-brats/results/brats_ablation_pooling.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:4
#SBATCH --time=30:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/waveformer-brats

# Execute the Python script
srun python 3_train.py