## **Nonstationarities project**


## Setup 

First download the GitHub repo and install as a pip package:

```
git clone https://github.com/guyhwilson/nonstationarities.git
pip install .
```



Requires Python 3 and anaconda/miniconda. To setup python virtual environment, use the following command from this level of directory:

`$ conda env create --file environment.yml`

`$ conda activate HMMrecal`

## Running 

First process a recent block of data: 

This outputs a file `TODO FILL IN`. Then pass this into the HMM recalibration code: 

`$ python HMMrecalibrate_VKF.py [file, str]`

## Running in Sherlock

Log into sherlock and run the following command to launch jupyter for the project:

`sbatch -p owners -t 14:00:00 -c 5 --mem=64gb /home/users/ghwilson/projects/nonstationarities/startJN.sh`

Then ssh into the compute node that it launches on:

`ssh $SLURM_NODELIST -L 30100:localhost:30100`

---------------------------

Session notes:
- one session has flipped pedestals (medial <--> lateral)
- 2021 T5 summer sessions have a delay period for some reason...?