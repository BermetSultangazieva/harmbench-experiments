#!/bin/bash
#SBATCH -J harmbench_eval
#SBATCH -A MLMI-bs816-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
source /rds/user/bs816/hpc-work/exp_diss/diss_env/bin/activate

export HF_HOME=/rds/user/bs816/hpc-work/hf_cache
export TRANSFORMERS_CACHE=/rds/user/bs816/hpc-work/hf_cache
export HF_HUB_CACHE=/rds/user/bs816/hpc-work/hf_cache/hub
export VLLM_USE_FLASHINFER_SAMPLER=0

HARMBENCH_DIR="/rds/user/bs816/hpc-work/exp_diss/harmbench-experiments"
MODEL_KEY="qwen3.5_9B_base_bon5"
METHOD="DirectRequest"

BEHAVIORS_PATH="$HARMBENCH_DIR/data/behavior_datasets/harmbench_behaviors_copyright_only.csv"
COMPLETIONS_PATH="$HARMBENCH_DIR/results/completions/$METHOD/$MODEL_KEY/completions.json"
SAVE_PATH="$HARMBENCH_DIR/results/eval/$METHOD/$MODEL_KEY/eval.json"

mkdir -p "$HARMBENCH_DIR/results/eval/$METHOD/$MODEL_KEY"
mkdir -p "$HARMBENCH_DIR/logs"

cd $HARMBENCH_DIR

JOBID=$SLURM_JOB_ID
echo "JobID: $JOBID"
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Model: $MODEL_KEY | Method: $METHOD"

python copyright_evaluate_completions_bon.py \
    --behaviors_path $BEHAVIORS_PATH \
    --completions_path $COMPLETIONS_PATH \
    --save_path $SAVE_PATH \
    > $HARMBENCH_DIR/logs/eval_${MODEL_KEY}_${METHOD}_${JOBID}.log 2>&1

echo "Done. $(date)"
