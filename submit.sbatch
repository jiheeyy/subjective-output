#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=logs/array_%A.out
#SBATCH --error=logs/array_%A.err
#SBATCH --array=1-1
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G #10G
#SBATCH --mail-type=ALL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jiheeyou@rcc.uchicago.edu  # mail notification for the job

conda activate disc
cd /home/jiheeyou/subjective-output

python experiment_main.py --directory rafdb_15k/DATASET/test/1