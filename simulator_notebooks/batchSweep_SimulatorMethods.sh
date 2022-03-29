#!/bin/sh
#SBATCH -p normal
#SBATCH --job-name="SimulatorMethods_sweep"
#SBATCH --begin=now
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=00-04:30:00
#SBATCH --array=0-49

ml load python/3.6.1 viz cudnn/7.6.4 cuda/10.0.130 py-scikit-learn/0.19.1_py36 py-matplotlib/3.1.1_py36  py-pandas/0.23.0_py36 py-sympy/1.1.1_py36 py-numpy/1.17.2_py36 py-jupyter/1.0.0_py36 system ffmpeg/4.2.1 py-numba/0.53.1_py36 py-scipy/1.4.1_py36

python3 /home/users/ghwilson/projects/nonstationarities/simulator_notebooks/optimize_methods.py --n_jobs 50 --jobID=$SLURM_ARRAY_TASK_ID --saveDir '/oak/stanford/groups/shenoy/ghwilson/nonstationarities/simulator/sweeps/'