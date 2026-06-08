#!/bin/bash
#SBATCH -J harmbench_completions
#SBATCH -A MLMI-bs816-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source /home/${USER}/.bashrc
source /rds/user/bs816/hpc-work/exp_diss/diss_env/bin/activate

export HF_HOME=/rds/user/bs816/hpc-work/hf_cache
export TRANSFORMERS_CACHE=/rds/user/bs816/hpc-work/hf_cache
export HF_HUB_CACHE=/rds/user/bs816/hpc-work/hf_cache/hub

HARMBENCH_DIR="/rds/user/bs816/hpc-work/exp_diss/HarmBench"
MODEL_KEY="llama3.2_3b_instruct"
METHOD="DirectRequest"

BEHAVIORS_PATH="$HARMBENCH_DIR/data/behavior_datasets/harmbench_behaviors_copyright_only.csv"
# TEST_CASES_PATH="$HARMBENCH_DIR/results/test_cases/$METHOD/$MODEL_KEY/test_cases.json"
TEST_CASES_PATH="$HARMBENCH_DIR/results/test_cases/DirectRequest/qwen2.5_3b_instruct/test_cases.json"
SAVE_PATH="$HARMBENCH_DIR/results/completions/$METHOD/$MODEL_KEY/completions.json"

mkdir -p "$HARMBENCH_DIR/results/completions/$METHOD/$MODEL_KEY"
mkdir -p "$HARMBENCH_DIR/logs"

cd $HARMBENCH_DIR

JOBID=$SLURM_JOB_ID
echo "JobID: $JOBID"
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Model: $MODEL_KEY | Method: $METHOD"

python generate_completions.py \
    --model_name $MODEL_KEY \
    --behaviors_path $BEHAVIORS_PATH \
    --test_cases_path $TEST_CASES_PATH \
    --save_path $SAVE_PATH \
    --max_new_tokens 512 \
    > $HARMBENCH_DIR/logs/completions_${MODEL_KEY}_${METHOD}_${JOBID}.log 2>&1

echo "Done. $(date)"
