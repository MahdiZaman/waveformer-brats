#!/bin/bash
#SBATCH --job-name=eval_brats_idwt_dec_v2
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/brats_idwt_inside/waveformer-brats/results/eval_brats_idwt_in_block.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/brats_idwt_inside/waveformer-brats/results/eval_brats_idwt_in_block.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=1:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/brats_idwt_inside/waveformer-brats

# Execute the Python script
srun python 5_compute_metrics.py --pred_name $PRED_NAME