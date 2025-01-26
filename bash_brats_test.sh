#!/bin/bash
#SBATCH --job-name=test_brats_idwt_dec_v2_wd_1e-5
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/brats_idwt_inside/waveformer-brats/results/test_brats_idwt_dec_v2_wd_1e-5.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/brats_idwt_inside/waveformer-brats/results/test_brats_idwt_dec_v2_wd_1e-5.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=2:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/brats_idwt_inside/waveformer-brats

# Execute the Python script
srun python 4_predict.py