#!/bin/bash
ml load python/3.6.1 viz cudnn/7.6.4 cuda/10.0.130 py-scikit-learn/0.19.1_py36 py-matplotlib/3.1.1_py36  py-pandas/0.23.0_py36 py-sympy/1.1.1_py36 py-numpy/1.17.2_py36 py-jupyter/1.0.0_py36 system ffmpeg/4.2.1 py-numba/0.53.1_py36 py-scipy/1.4.1_py36 py-schwimmbad/0.3.1_py36

jupyter-notebook --no-browser --port=30100 --notebook-dir= .