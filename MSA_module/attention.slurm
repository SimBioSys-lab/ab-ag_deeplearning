#!/bin/bash
#SBATCH -J PT_test
#SBATCH -o PT_test.out
#SBATCH -e PT_test.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -t 8:00:00
#SBATCH --export=ALL
#SBATCH --mem=150000
#SBATCH --cpus-per-task=10
#SBATCH --mail-user=xing.liu@northeastern.edu
#SBATCH --mail-type=ALL

module load cuda/12.1
#python msa_training.py
#python missing_res_pred.py
#python interface_pred.py
#python SASA_train.py
#python SS_train.py
python PT_test.py
#python read_para.py
