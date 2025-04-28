## Installation

First install all python packages (tested on Python 3.9):  

pip install -r requirements.txt  

If cupy does not install, it can be removed from requirements.txt if you just want to run the main figure notebooks (see below). Then install this repository as a python package:  

pip install -e .

## Main figures

Notebooks that reproduce the main figures are as follows in the nonstationarities repository (https://github.com/guyhwilson/nonstationarities):  

Figure 1 - Nonstationarity characterization (notebooks/Figure1_NeuralDriftCharacteristics.ipynb)
Figure 2 - Stabilizer & HMM visualization (notebooks/Fig2_ExampleApproaches.ipynb)  
Figure 3 - Offline methods comparison on session pairs (notebooks/Fig_3_and_Supp_Figs_3_4.ipynb)  
Figure 4 - Comparison of recalibration strategies in simulation (simulator_notebooks/Fig_4a_4b.ipynb, simulator_notebooks/Fig_4c.ipynb, simulator_notebooks/Fig_4d.ipynb, simulator_notebooks/Fig_4e.ipynb)  
Figure 5 - Closed-loop performance comparison in T5 (notebooks/Fig_5.ipynb)  
Figure 6 - Offline evaluation of  PRI-T and RTI on freeform, personal use data (notebooks/Fig_6.ipynb; but note that T11 personal use data was not able to be released becuase it may contain PHI, so this cannot be run)  

Before running these notebooks on your computer, change DATA_DIR and FIG_DIR in utils/preprocessing/preprocess.py to point to the unzipped, downloaded data directory and an empty directory where figures should be saved. Also note that the Figure 3 notebook depends on decoding analysis results created by the Figure 1 notebook, so run that one first. 

## Data formatting

There are a few different data sources used in this repo.

### T5 closed-loop cursor control datasets (retrospectively analyzed offline)

These are saved as .mat files in the T5/raw_data/historical folder (sessions 2016.09.26 - 2020.02.26) and T5/raw_data/new folder (2021.04.26 - 2021.07.26), and were originally intended for loading using MATLAB. To obtain minimally processed results using scipy, access the data via:

dat = scipy.io.loadmat("/path/to/mat_file.mat")["dataset"][0][0]

The variable "dat" is now a list with entries:

dat[0][0] (list[str]) - integer labels for each block  
dat[1] (list[float]) - 2D target size at each timestep  
dat[2] (list[float]) - 2D cursor size at each timestep  
dat[3] (list[int]) - corresponding block number for each timestep  
dat[4] (list[float]) - system clock  
dat[5] (list[float]) - clock time on NSPs  
dat[6] (list[float]) - 2D cursor position at each timestep  
dat[7] (list[float]) - 2D target position at each timestep  
dat[8] (list[int]) - indicator at each timestep for if cursor is on target  
dat[10] (list[float]) - 2D decoded velocity at each timestep  
dat[12] (list[int]) - 192D array of binned firing rates at each timestep  
dat[13] (list[float]) - RMS threshold for spike detection  
dat[15] (2D float) - trial start and stop indices in MATLAB convention (1-indexing, inclusive end range)  
dat[17] (list[int]) - active trial times  
dat[18] (list[str]) - list of block names  
dat[19] (list[int]) - trial success indicator (1 = success, 0 = failed)  

To obtain more nicely formatted data in Python, you can use the DataStruct object in the accompanying repository. For example:

from utils.preprocessing.preprocess import DataStruct  
dat = DataStruct("/path/to/mat_file.mat")  

### T5 online recalibration experiment datasets

These data are stored as individual blocks for each closed-loop experiment in:

T5/raw_data/one_month_recal/t5.YYYY.MM.DD/Data/TaskData/FittsTask/YYYY-MM-DD_block_Z.pkl

Each pickle file has a "task_df" field containing a pandas dataframe where each row corresponds to a timepoint in the recording. Columns are:

target_pos (2D float) - position of the target on the screen at each timestep  
cursor_pos (2D float) - position of the cursor on the screen  
cursor_vel (2D float) - instantaneous cursor velocity  
cursor_color (RGB int tuple) - color of the cursor  
target_color (RGB int tuple) - color of the target  
state (int) - an integer-valued game state indicator. 0 -- regular game play, 1 -- hovering over target, 2 -- success, 3 -- failure.  
trial_counter (float) - UNIX timestamp (UTC) of elapsed trial time; unused  
hold_counter (float) - UNIX timestamp (UTC) of elapsed hold time; unused  
bias_killer (2D float) - bias correction state  
alpha (float) - exponential smoothing value  
beta (float) - gain value  
max_trial_time_sec (float) - system's max allowed trial time  
hold_time_sec (float) - hold time needed for dwell-based selection of target  
target_radius_options (float, list[float]) - target radius size options; chosen randomly on each new trial  
target_radius (float) - radius of the current trial's target  
xPC_clock (float) clock times as measured on xPC system  
neural (192-D float) - binned spike counts at each timestep from channels  
raw_vel (2D float) - the raw velocity signal as decoded, prior to exponential smoothing and gain adjustment  
task_clock (byte str) - clock time on system used for task handling in seconds  


### Recalibration technique sweep results

Results are stored for each technique in:

T5/XXX/test/scores_ID_YY.npy

where XXX corresponds to a specific method (e.g. "ADAN"). The .npy files are generated from a parallel compute job and should be combined to return the full sweep results for a given method.

Each such file is a list of dictionaries, with individual entries having key-value pairs:

file (str) - corresponds to which pair of files was used for measuring recalibration performance (reference and new sessions)  
R2_score (float) - R2 performance on test split from the new session  
r2_score (float) - pearson correlation of predictions with intention vector from new session  
days_apart (int) - number of days apart between reference and new session  
norecal_R2_score (float) - R2 score on new day without recalibration  
norecal_pearson_r (float) - pearson correlation on new day without recalibration  
meanrecal_R2_score (float) - R2 score on new day with mean adaptation  
meanrecal_pearson_r (float) - pearson correlation on new day with mean adaptation  
suprecal_R2_score (float) - R2 score on new day with supervised retraining  
suprecal_pearson_r (float) - pearson correlation on new day with supervised retraining  

Additionally, the hyperparameters used for the recalibration on these specific data are listed in additional entries, "batch_size" (int), or "n_components" (int). In all cases these are singletons of type float, int, or string.

### Simulator sweeps

Results for various sweeps are stored as .npy files at:

simulator/performance/regular/ (main results from Fig. 4A and B)  
simulator/performance/efficiency/ (efficiency sweeps from Fig. 4C)  
simulator/performance/SNR/ (SNR sweeps from Fig. 4D)  
simulator/performance/instability_analysis/ (Fig. 4E)  

For the main results, each .npy file is a Time x Methods array of floats containing the average trial time for each recalibration method on each day. There are 61 days and 8 methods tested (ordering: 'Gain', 'Supervised', 'PRI-T', 'PRI-T (static)', 'Stabilizer', 'Stabilizer (static)', 'RTI', 'RTI (static)').

For efficiency and SNR sweeps, each .npy contains a dictionary of the results from a single simulator run, with key-value pairs:

method (str) - recalibration method used  
ttt (float array) - individual trial times in seconds  
neuralTuning (3D float) - channels x 3 of mean FR and XY encoding weights  

For the efficiency sweeps, there is also a field "nSimSteps' (int) corresponding to the length of each simulation in steps. Both file groups also contain hyperparameters for each recalibration method used.

For the instability analysis in Fig. 4E, the .npy files are dictionaries containing data from simulation runs after a critical error threshold has been passed, and are of the form:

ttt (list[float]) - list of trial times on threshold passing day  
stabilizers (object) - singleton list of Stabilizer objects  
ss_decoder_dicts (dict) - singleton of corresponding dictionary containing stabilizer config  
n_days_to_threshold (list[int]) - singleton; number of days until threshold passed  
corrvals (list[float]) - angular error between linear kinematics decoder and encoding subspace  
new_cfgs (list[dict]) - singleton containing simulator state on threshold passing day  
