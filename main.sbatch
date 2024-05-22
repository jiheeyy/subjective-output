#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G #10G
#SBATCH --mail-type=END  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jiheeyou@rcc.uchicago.edu  # mail notification for the job

source /opt/conda/etc/profile.d/conda.sh
conda activate disc

cd /home/jiheeyou/subjective-output
python main.py --directory rafdb_15k/DATASET/train/6
python main.py --directory rafdb_15k/DATASET/train/7
#test_set
#rafdb_15k/DATASET/test/1