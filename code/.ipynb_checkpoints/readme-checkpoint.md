## To-Do list


*Clean-up*
- DONE: remove `get_ClickSessions()` and replace with `get_Sessions(getClick = True)`  
- combine/update old notebooks and add comments for walkthroughs


*Simulator*

- DONE: implement click signal
	- scalar value in orthogonal dimension
	- just have as fxn of distance to target
- get simulator click decoder training working
- implement click decoder cooldown period 
- use HMM or logistic regression? 
- implement velocity encoding dimension (see Frank unpublished paper)


*T5 dataset*
- do another check on bad performance days



*Methods*
- DONE: refactor `hmm.py` code to take argument `adjustKappa()` for downweighting near-target samples
	- run T5 analysis to optimize this 
    
    
*Decoding*
- fix `train_test_supervised` lr model to have `fit_intercept = True`
- build out RNN decoder codebase
- look at HMM recalibration for RNN 
	
	
	
*Comparisons*

- optimize T5 HMM vs subspace stabilizer comparison
