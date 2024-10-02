#!/bin/bash
#SBATCH -J msa_attention
#SBATCH -o msa_attention.out
#SBATCH -e msa_attention.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -t 02:00:00
##SBATCH --export=ALL
#SBATCH --mem=150000

module load cuda/12.1
python msa_training.py
