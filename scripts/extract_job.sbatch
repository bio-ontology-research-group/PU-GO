#!/bin/bash --login
#SBATCH --time=3-00:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --reservation=A100
#SBATCH --mem=64G
#SBATCH --job-name=esm

source ~/anaconda3/bin/activate
conda activate deepgo

python scripts/extract.py esm2_t48_15B_UR50D esm_missing.fa esm15B_miss --repr_layers 48 --include mean
