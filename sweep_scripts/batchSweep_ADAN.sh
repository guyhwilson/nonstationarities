#!/bin/sh
#SBATCH -p owners,shenoy,shauld
#SBATCH --job-name="ADAN_sweep"
#SBATCH --begin=now
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00-5:00:00
#SBATCH --array=0-809

ml load python/3.9.0 cudnn/8.1.1.33 cuda/11.2.0

#python3 /home/users/ghwilson/projects/nonstationarities/notebooks/optimizeADANWithinDay.py --jobID=$SLURM_ARRAY_TASK_ID

# TRAIN
python3 /home/users/ghwilson/projects/nonstationarities/sweep_scripts/optimizeADAN.py --participant=T5 --n_jobs=810 --jobID=$SLURM_ARRAY_TASK_ID --saveDir='/oak/stanford/groups/shenoy/ghwilson/nonstationarities/T5/ADAN/ADANsweep2/'

# TEST
#python3 /home/users/ghwilson/projects/nonstationarities/sweep_scripts/optimizeADAN.py --participant=T5 --n_jobs=1000 --jobID=$SLURM_ARRAY_TASK_ID --saveDir='/oak/stanford/groups/shenoy/ghwilson/nonstationarities/T5/ADAN/test/'