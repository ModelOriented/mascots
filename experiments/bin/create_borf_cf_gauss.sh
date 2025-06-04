#!/bin/bash
#SBATCH --account=mi2lab-normal
#SBATCH --job-name=borf-borf-gauss
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dawid.pludowski@gmail.com
#SBATCH --output=/mnt/evafs/groups/mi2lab/dpludowski/code/borf-xai/log/create_borf_%A_%a.log

. /mnt/evafs/groups/mi2lab/dpludowski/miniconda3/etc/profile.d/conda.sh

conda activate borf
export PYTHONPATH=`pwd`
export TASK_ID=${SLURM_ARRAY_TASK_ID}
export SWAP_METHOD='gaussian'

python experiments/bin/create_borf_cf.py

# how to run
# sbatch --array=0-12 experiments/bin/create_borf_cf_gauss.sh