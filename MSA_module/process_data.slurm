#!/bin/bash
#SBATCH -J data_pre
#SBATCH -o data_pre.out
#SBATCH -e data_pre.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -t 08:00:00
##SBATCH --export=ALL
##SBATCH --mem=100000
##SBATCH --cpus-per-task=20

module load cuda/12.1
#python msa_training.py
#python missing_res_pred.py
#python interface_pred.py
python data_preprocessing_mask.py
