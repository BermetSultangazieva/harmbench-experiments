#!/bin/bash
#SBATCH -J harmbench_bon5_directrequest
#SBATCH -A MLMI-bs816-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
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

# ---- vLLM speed note ----------------------------------------------------------
# vLLM's throughput comes from PagedAttention + continuous batching + the FLASH_ATTN
# backend (all prebuilt, already active). FlashInfer is ONLY used for the final
# top-p/top-k sampling step, which is a negligible fraction of runtime.
#
# FlashInfer JIT-compiles its sampler with nvcc, but `module load rhel8/default-amp`
# provides CUDA 11.4 and FlashInfer needs CUDA >= 12 -> the build fails and the engine
# never starts. Disabling the FlashInfer *sampler* below makes vLLM use its native
# PyTorch sampler instead: this costs ~0% throughput (attention is unchanged).
export VLLM_USE_FLASHINFER_SAMPLER=0

# ---- OPTIONAL: keep FlashInfer's sampler (only if a CUDA >=12 toolkit is available) ----
# Run `module avail cuda` on the cluster; if you see e.g. cuda/12.x, comment out the
# line above and uncomment + fix the module name below so FlashInfer can compile once:
# module load cuda/12.4          # <-- replace with the actual CUDA>=12 module name
# export CUDA_HOME=$CUDA_INSTALL_PATH

HARMBENCH_DIR="/rds/user/bs816/hpc-work/exp_diss/harmbench-experiments"

# ========== Parameters ==========
MODEL_KEY="qwen3.5_9B_base"   # <-- change to the target model you want
METHOD="DirectRequest"

# Best-of-N sampling settings
NUM_COMPLETIONS=5                  # 5 stochastic completions per test case
TEMPERATURE=0.8                    # >0 required for stochastic sampling
TOP_P=0.9

BEHAVIORS_PATH="$HARMBENCH_DIR/data/behavior_datasets/harmbench_behaviors_copyright_only.csv"

# Existing DirectRequest test cases (behavior strings; model-agnostic)
TEST_CASES_PATH="$HARMBENCH_DIR/results/test_cases/DirectRequest/qwen2.5_3b_instruct/test_cases.json"

# Saved to a BoN-specific folder so the existing greedy DirectRequest completions are NOT overwritten
SAVE_PATH="$HARMBENCH_DIR/results/completions/$METHOD/${MODEL_KEY}_bon${NUM_COMPLETIONS}/completions.json"

mkdir -p "$(dirname "$SAVE_PATH")"
mkdir -p "$HARMBENCH_DIR/logs"

cd "$HARMBENCH_DIR"

JOBID=$SLURM_JOB_ID
echo "======================================="
echo "JobID: $JOBID"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Model: $MODEL_KEY | Method: $METHOD"
echo "Best-of-N: n=$NUM_COMPLETIONS temperature=$TEMPERATURE top_p=$TOP_P"
echo "======================================="

nvidia-smi

python generate_completions_bon.py \
    --model_name $MODEL_KEY \
    --behaviors_path $BEHAVIORS_PATH \
    --test_cases_path $TEST_CASES_PATH \
    --save_path $SAVE_PATH \
    --max_new_tokens 512 \
    --num_completions $NUM_COMPLETIONS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --generate_with_vllm \
    > $HARMBENCH_DIR/logs/completions_${MODEL_KEY}_${METHOD}_bon${NUM_COMPLETIONS}_${JOBID}.log 2>&1

echo "Finished: $(date)"
