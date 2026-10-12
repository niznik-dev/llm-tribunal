#!/bin/bash
# Example SLURM batch script for a model too big for one GPU: all four GPUs on one node.
# Same as job.slurm, but with --device-map auto, which places consecutive layers on
# consecutive GPUs. The GPUs take turns rather than working in parallel, so four GPUs buy
# room for the model, not speed.
#
# Everything after the script name goes straight to run.py:
#     sbatch job-multigpu.slurm --model /path/to/Qwen3.5-122B-A10B --limit 3
#
# Sizing: bf16 weights take 2 bytes per parameter, so a 122B model needs ~233 GiB of GPU
# memory; 4 x 80 GB A100s hold 320 GiB, leaving room for activations and the KV cache.
#
# Belong to more than one SLURM account? Some clusters (della) then require choosing one:
#     sbatch --account=<account> job-multigpu.slurm ...
#
# Compute nodes have no internet: download weights first, on a login node (download.py).

#SBATCH --job-name=llm-tribunal-multigpu
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G                # host RAM: a quarter of a della 4-GPU node, per the usual fair share
#SBATCH --gres=gpu:4              # GPU count; della picks the partition from this
#SBATCH --constraint=gpu80        # della: 80 GB A100s only
#SBATCH --time=01:00:00           # loading ~233 GiB from disk alone takes several minutes

module purge
module load anaconda3/2026.7
conda activate llm-tribunal

# Where download.py put the weights. Point the cache at scratch, not home, which is small.
# Unset here, the job inherits it from your shell, or falls back to ./models.
# export HF_HUB_CACHE=/scratch/gpfs/<group>/<you>/hf-cache

# What the job actually got, for the log.
echo "job-multigpu.slurm: $(hostname), job $SLURM_JOB_ID, GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

python run.py --device-map auto "$@"
status=$?
# A status line to check for: run.py's output is model text and can contain any word.
echo "job-multigpu.slurm: run.py exited with $status"
exit $status
