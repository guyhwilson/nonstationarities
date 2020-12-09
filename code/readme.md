{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## To-Do list\n",
    "\n",
    "*prep_HMMData()*\n",
    "- join adjacent trials so that the returned list contains contiguous segments\n",
    "    - figure out if returning individual trial lists to train_HMMRecalibrate causes bad performance (linear reg\n",
    "      models show bad performance because the time snippets are so short)\n",
    "    - maybe trash and just add optional return_cursorPos parameter to getTrainTest()\n",
    "\n",
    "*simulator*\n",
    "- get multi-session repeated recalibration working\n",
    "- switch over from as-is original HMM and linear regression training code to HMM recal code from T5 offline data\n",
    "    - e.g. use train_HMMRecalibrate() in place of current code chunks"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Speech",
   "language": "python",
   "name": "speech"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
