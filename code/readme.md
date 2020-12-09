## To-Do list

*prep_HMMData()*
- join adjacent trials so that the returned list contains contiguous segments
    - figure out if returning individual trial lists to train_HMMRecalibrate causes bad performance (linear reg
      models show bad performance because the time snippets are so short)
    - maybe trash and just add optional return_cursorPos parameter to getTrainTest()

*simulator*
- get multi-session repeated recalibration working
- switch over from as-is original HMM and linear regression training code to HMM recal code from T5 offline data
    - e.g. use train_HMMRecalibrate() in place of current code chunks