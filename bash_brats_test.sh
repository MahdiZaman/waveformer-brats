#!/bin/bash
#SBATCH --job-name=test_brats_idwt_dec_v2
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/brats_idwt_inside/waveformer-brats/results/test_brats_idwt_in_block.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/brats_idwt_inside/waveformer-brats/results/test_brats_idwt_in_block.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:2
#SBATCH --time=48:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/brats_idwt_inside/waveformer-brats

# Execute the Python script
srun python 4_predict.py