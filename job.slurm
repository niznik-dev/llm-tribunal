#!/bin/bash
# Example SLURM batch script: ask a model the question bank on a cluster GPU.
#
# Everything after the script name goes straight to run.py:
#     sbatch job.slurm --model /path/to/Qwen3.8-27B --limit 3
#     sbatch job.slurm --model Qwen/Qwen3.5-9B --samples 5 --temperature 0.7
#
# #SBATCH lines are defaults; flags given to sbatch override them, e.g. a model too big
# for one GPU (also pass --device-map auto so run.py splits it across them):
#     sbatch --gres=gpu:2 --time=2:00:00 job.slurm --model <70B> --device-map auto
#
# Belong to more than one SLURM account? Some clusters (della) then require choosing one:
#     sbatch --account=<account> job.slurm ...
#
# Compute nodes have no internet: download weights first, on a login node (download.py).

#SBATCH --job-name=llm-tribunal
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                 # host RAM: weights stream onto the GPU; 27B peaked under 2 GB
#SBATCH --gres=gpu:1              # GPU count; della picks the partition from this
#SBATCH --constraint=gpu80        # della: 80 GB A100s only (bf16 27B needs ~54 GB)
#SBATCH --time=00:30:00

module purge
module load anaconda3/2026.7
conda activate llm-tribunal

# Where download.py put the weights. Point the cache at scratch, not home, which is small.
# Unset here, the job inherits it from your shell, or falls back to ./models.
# export HF_HUB_CACHE=/scratch/gpfs/<group>/<you>/hf-cache

# What the job actually got, for the log.
echo "job.slurm: $(hostname), job $SLURM_JOB_ID, GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python run.py "$@"
status=$?
# A status line to check for: run.py's output is model text and can contain any word.
echo "job.slurm: run.py exited with $status"
exit $status
