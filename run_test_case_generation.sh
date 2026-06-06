#!/bin/bash
#SBATCH -J harmbench_zeroshot_testcases
#SBATCH -A MLMI-bs816-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source /rds/user/bs816/hpc-work/exp_diss/diss_env/bin/activate

HARMBENCH_DIR="/rds/user/bs816/hpc-work/exp_diss/HarmBench"

# ========== Parameters ==========
METHOD_NAME="ZeroShot"
EXPERIMENT_NAME="ZeroShot"
BEHAVIORS_PATH="$HARMBENCH_DIR/data/behavior_datasets/harmbench_behaviors_text_all.csv"
SAVE_DIR="$HARMBENCH_DIR/results/test_cases/ZeroShot/copyright_all"
METHOD_CONFIG_FILE="$HARMBENCH_DIR/configs/method_configs/ZeroShot_config.yaml"

# Optional parameters (leave empty to skip)
START_IDX=""
END_IDX=""
RUN_ID=""
OVERWRITE=""
VERBOSE=""

mkdir -p "$SAVE_DIR"
mkdir -p "$HARMBENCH_DIR/logs"

cd $HARMBENCH_DIR

JOBID=$SLURM_JOB_ID
echo "JobID: $JOBID"
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Method: $METHOD_NAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ========== Run generate_test_cases.py ==========
python -u generate_test_cases.py \
    --method_name $METHOD_NAME \
    --method_config_file $METHOD_CONFIG_FILE \
    --experiment_name $EXPERIMENT_NAME \
    --behaviors_path $BEHAVIORS_PATH \
    --save_dir $SAVE_DIR \
    --overwrite \
    $([ ! -z "$START_IDX" ] && echo "--behavior_start_idx $START_IDX") \
    $([ ! -z "$END_IDX" ] && echo "--behavior_end_idx $END_IDX") \
    $([ ! -z "$RUN_ID" ] && echo "--run_id $RUN_ID") \
    $([ "$OVERWRITE" == "True" ] && echo "--overwrite") \
    $([ "$VERBOSE" == "True" ] && echo "--verbose") \
    > $HARMBENCH_DIR/logs/testcases_${METHOD_NAME}_${JOBID}.log 2>&1

echo "Done. $(date)"
