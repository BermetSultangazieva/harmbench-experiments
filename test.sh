#!/bin/bash
#SBATCH -J cuda_test
#SBATCH -A MLMI-bs816-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source /rds/user/bs816/hpc-work/exp_diss/diss_env/bin/activate

nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch:', torch.__version__)"
