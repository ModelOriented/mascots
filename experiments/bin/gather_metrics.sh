conda activate borf

export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=-1

python experiments/bin/evaluate_results_borf.py
python experiments/bin/evaluate_results_mcels.py
python experiments/bin/evaluate_results_glacier.py

python experiments/bin/create_summary.py