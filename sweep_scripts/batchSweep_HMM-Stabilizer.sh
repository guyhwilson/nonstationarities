#!/bin/sh
#SBATCH -p normal,owners
#SBATCH --job-name="HMMStabilizer_paramSweep"
#SBATCH --begin=now
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=00-5:00:00
#SBATCH --array=0-299

ml load python/3.9.0 cudnn/8.1.1.33 cuda/11.2.0

#python3 /home/users/ghwilson/projects/nonstationarities/sweep_scripts/optimizeHMM-Stabilizer.py --participant=T5 --n_jobs=100 --jobID=$SLURM_ARRAY_TASK_ID --saveDir='/oak/stanford/groups/shenoy/ghwilson/nonstationarities/T5/HMM-Stabilizer/HMMStabilizersweep/'

python3 /home/users/ghwilson/projects/nonstationarities/sweep_scripts/test_HMM-Stabilizer.py --participant=T5 --n_jobs=300 --jobID=$SLURM_ARRAY_TASK_ID --saveDir='/oak/stanford/groups/shenoy/ghwilson/nonstationarities/T5/HMM-Stabilizer/test/'
