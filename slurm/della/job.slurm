#!/bin/bash
#SBATCH --job-name=llm-tribunal     # create a short name for your job
#SBATCH --output=logs/%x-%j.out     # stdout file (logs/ must exist; %x = job name, %j = job ID)
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                   # memory per node; weights stream to the GPU, 27B peaked < 2 GB
#SBATCH --gres=gpu:1                # number of gpus per node
#SBATCH --constraint=gpu80          # della: 80 GB A100s only (bf16 27B needs ~54 GB)
#SBATCH --time=00:30:00             # total run time limit (HH:MM:SS)
##SBATCH --mail-type=begin,end,fail # receive email notifications (remove one # to enable)
##SBATCH --mail-user=<YourNetID>@princeton.edu

# Ask a model the question bank on one GPU. From the repo root; arguments go to run.py:
#     sbatch slurm/della/job.slurm --model /path/to/Qwen3.8-27B --limit 3
# More options (accounts, bigger models, the weights cache): README, "Running on a cluster".

module purge
module load anaconda3/2026.7
conda activate llm-tribunal

# Compute nodes have no internet, so point this at wherever download.py put the weights.
# export HF_HUB_CACHE=/scratch/gpfs/<group>/<you>/hf-cache

echo "job.slurm: $(hostname), job $SLURM_JOB_ID, GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python run.py "$@"
status=$?
# A status line to check for: run.py's output is model text and can contain any word.
echo "job.slurm: run.py exited with $status"
exit $status
