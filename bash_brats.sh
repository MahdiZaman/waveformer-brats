#!/bin/bash
#SBATCH --job-name=train_brats_idwt_dec
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/waveformer-brats/results/brats_1111_idwt_dec.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/waveformer-brats/results/brats_1111_idwt_dec.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:2
#SBATCH --time=48:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/waveformer-brats

# Execute the Python script
srun python 3_train.py