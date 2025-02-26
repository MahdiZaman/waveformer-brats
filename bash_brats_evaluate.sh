#!/bin/bash
#SBATCH --job-name=eval_brats_residual_up_simple_ref
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/waveformer-brats/results/eval_brats_residual_up_simple_ref.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/waveformer-brats/results/eval_brats_residual_up_simple_ref.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=2:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/waveformer-brats

# Execute the Python script
srun python 5_compute_metrics.py --pred_name $PRED_NAME