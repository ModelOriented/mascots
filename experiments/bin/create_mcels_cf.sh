#!/bin/bash
#SBATCH --account=mi2lab-normal
#SBATCH --job-name=borf-mcels
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dawid.pludowski@gmail.com
#SBATCH --output=/mnt/evafs/groups/mi2lab/dpludowski/code/borf-xai/log/create_mcels_%A_%a.log

. /mnt/evafs/groups/mi2lab/dpludowski/miniconda3/etc/profile.d/conda.sh

conda activate borf
export PYTHONPATH=`pwd`
export TASK_ID=${SLURM_ARRAY_TASK_ID}

python experiments/bin/create_mcels_cf.py \
     --run_mode turing \
    --jobs_per_task 10 --samples_per_task 50 \
    --algo cf --seed_value 1 \
    --enable_lr_decay False --background_data train \
    --background_data_perc 100 --enable_seed True \
    --max_itr 1000 --run_id 0 --bbm dnn --enable_tvnorm False \
    --enable_budget False --dataset_type test --l_budget_coeff 1.0 \
    --run 1 --l_tv_norm_coeff 1.0 --l_max_coeff 1.0

# how to run
# sbatch --array=0-12 experiments/bin/create_mcels_cf.sh