#!/bin/bash
#SBATCH -J test_attention
#SBATCH -o test_attention.out
#SBATCH -e test_attention.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -t 08:00:00
##SBATCH --export=ALL

#module load cuda/11.3
python attention.py
