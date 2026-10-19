#!/bin/bash
#SBATCH --job-name=llm-tribunal-multigpu  # create a short name for your job
#SBATCH --output=slurm-%x-%j.out          # stdout file (%x = job name, %j = job ID)
#SBATCH --nodes=1                         # node count
#SBATCH --ntasks=1                        # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                         # memory per node; weights stream to the GPUs, 122B peaked < 2 GB
#SBATCH --gres=gpu:4                      # number of gpus per node
#SBATCH --constraint=gpu80                # della: 80 GB A100s only
#SBATCH --time=01:00:00                   # total run time limit (HH:MM:SS); loading 233 GiB takes minutes
##SBATCH --mail-type=begin,end,fail       # receive email notifications (remove one # to enable)
##SBATCH --mail-user=<YourNetID>@princeton.edu

# Like job.slurm, but splits a model too big for one GPU across four (--device-map auto):
#     sbatch job-multigpu.slurm --model /path/to/Qwen3.5-122B-A10B --limit 3
# Why four GPUs, and why they add room but not speed: README, "Running on a cluster".

module purge
module load anaconda3/2026.7
conda activate llm-tribunal

# Compute nodes have no internet, so point this at wherever download.py put the weights.
# export HF_HUB_CACHE=/scratch/gpfs/<group>/<you>/hf-cache

echo "job-multigpu.slurm: $(hostname), job $SLURM_JOB_ID, GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

python run.py --device-map auto "$@"
status=$?
# A status line to check for: run.py's output is model text and can contain any word.
echo "job-multigpu.slurm: run.py exited with $status"
exit $status
