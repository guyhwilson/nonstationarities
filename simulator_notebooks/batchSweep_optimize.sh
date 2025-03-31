#!/bin/sh
#SBATCH -p normal,owners,henderj,shauld
#SBATCH --job-name="SimulatorMethods_sweep"
#SBATCH --begin=now
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=00-05:30:00
#SBATCH --array=0-593

ml load python/3.9.0 cudnn/8.1.1.33 cuda/11.2.0 py-pytorch/1.8.1_py39

python3 /home/users/ghwilson/projects/nonstationarities/simulator_notebooks/optimize_methods.py --n_jobs 594 --jobID=$SLURM_ARRAY_TASK_ID --saveDir '/oak/stanford/groups/henderj/ghwilson/nonstationarities/simulator/HP_sweeps/regular/'