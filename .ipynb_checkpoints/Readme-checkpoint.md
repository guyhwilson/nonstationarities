## **Nonstationarities project**

## Setup 
Requires Python 3 and anaconda/miniconda. To setup python virtual environment, use the following command from this level of directory:

`$ conda env create --file environment.yml`

`$ conda activate HMMrecal`

## Running 

First process a recent block of data: 

This outputs a file `TODO FILL IN`. Then pass this into the HMM recalibration code: 

`$ python HMMrecalibrate_VKF.py [file, str]`





---------------------------

TODO:
- move mean subtraction out of models and evaluation functions and into preprocessing



#### HMM code + repo restructuring update
 
List of notebooks:
- `click_BehaviorT5`
- `click_HMM_T5`
- `comparisons_all`
- `CursorDecoding_LinearRegression` 
- `DecodingClick_T5`
- `DEVELOPMENT`
- `example_Behavior`
- `example_MeanRecalibration` 
- `example_RTI_Recalibration`
- `example_SubspaceRealignment` 
- `example_T5_vanillaHMM`
- `optimize_vanillaHMM`
- `optimize_SubspaceRealignment` 
- `PDs_AcrossSession`
- `PDs_FlexibleVonMises`
- `PDs_WithinSession`
- `preprocessing_FindBadSessions`
- `preprocessing_ScreenRealignments`
- `prob_weighting_HMM`
- `SNR_T5` 
- `stability_MeanRecalibration`


#### *Regularized linear regression*

Current velocity decoders can be overfit - try using ridge regression and redoing relevant analyses. 
