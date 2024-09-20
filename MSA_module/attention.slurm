#!/bin/bash
#SBATCH -J test_attention
#SBATCH -o test_attention.out
#SBATCH -e test_attention.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -t 00:10:00
##SBATCH --export=ALL

module load cuda/12.1
python msa_training.py
