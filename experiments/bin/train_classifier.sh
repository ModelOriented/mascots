#!/bin/bash
#SBATCH --account=mi2lab-normal
#SBATCH --job-name=borf-classifier
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dawid.pludowski@gmail.com
#SBATCH --output=/mnt/evafs/groups/mi2lab/dpludowski/code/borf-xai/log/train-classifier_%A_%a.log

. /mnt/evafs/groups/mi2lab/dpludowski/miniconda3/etc/profile.d/conda.sh

conda activate borf
export PYTHONPATH=`pwd`
export TASK_ID=${SLURM_ARRAY_TASK_ID}

python experiments/bin/train_classifier.py

conda activate mach-ext
export PYTHONPATH=`pwd`
export TASK_ID=${SLURM_ARRAY_TASK_ID}

python experiments/bin/train_classifier.py

# how to run
# sbatch --array=0-12 experiments/bin/train_classifier.sh