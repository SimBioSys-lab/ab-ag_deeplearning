#!/bin/bash
#SBATCH -J PT_2l
#SBATCH -o PT_2l.out
#SBATCH -e PT_2l.err
#SBATCH -p multigpu
##SBATCH --reservation=xing.liu_test
#SBATCH --gres=gpu:v100-sxm2:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH --export=ALL
#SBATCH --mem=150000
#SBATCH --cpus-per-task=20
#SBATCH --mail-user=xing.liu@northeastern.edu
#SBATCH --mail-type=ALL

module load cuda/12.1
#python msa_training.py
#python missing_res_pred.py
#python interface_pred.py
#python SASA_train.py
#python SS_train.py
python PT_train_val.py
#sbatch SASA.slurm
