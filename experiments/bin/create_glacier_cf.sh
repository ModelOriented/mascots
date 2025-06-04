#!/bin/bash
#SBATCH --account=mi2lab-normal
#SBATCH --job-name=borf-glacier
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dawid.pludowski@gmail.com
#SBATCH --output=/mnt/evafs/groups/mi2lab/dpludowski/code/borf-xai/log/create_glacier_%A_%a.log

. /mnt/evafs/groups/mi2lab/dpludowski/miniconda3/etc/profile.d/conda.sh

conda activate mach-ext
export PYTHONPATH=`pwd`
export TASK_ID=${SLURM_ARRAY_TASK_ID}

python experiments/bin/create_glacier_cf.py

# how to run
# sbatch --array=0-5 experiments/bin/create_glacier_cf.sh