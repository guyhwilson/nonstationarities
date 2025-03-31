#!/bin/sh
#SBATCH -p normal,owners,henderj,shauld
#SBATCH --job-name="Sim_performance_sweep"
#SBATCH --begin=now
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=00-05:30:00
#SBATCH --array=0-399


ml load python/3.9.0 py-tensorflow/2.9.1_py39
source /oak/stanford/groups/henderj/ghwilson/miniconda3/etc/profile.d/conda.sh
conda activate nonstationarities

python /home/users/ghwilson/projects/nonstationarities/simulator_notebooks/performance_sweep.py --n_jobs 400 --jobID=$SLURM_ARRAY_TASK_ID --saveDir '/oak/stanford/groups/henderj/ghwilson/nonstationarities/simulator/performance/regular2/'