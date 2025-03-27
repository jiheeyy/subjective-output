#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --array=1-1
#SBATCH --time=2:00:00
#SBATCH --constraint=a100
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jiheeyou@rcc.uchicago.edu  # mail notification for the job


# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "My SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

job_id=$SLURM_ARRAY_JOB_ID
source /home/jiheeyou/miniforge3/etc/profile.d/conda.sh
conda activate disc

result_file="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "result file is ${result_file}"

cd /home/jiheeyou/subjective-output
python EMD_resnet_train.py --epoch 100 --var $1 --attribute $2 --transform resnet

# string, string