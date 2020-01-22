# AttentiveFP comes from github: https://github.com/OpenDrugAI/AttentiveFP/blob/master/code/1_POC_test_Solubility.ipynb


## In the prediction of the ESOL, the author splitted the dataset to training valid, and test set by a random seed of 108 after remove one uncovered compound, then the author use early-stopping strategy by monitor the mean squared error(MSE) of valid Set, in their paper, they got the RMSE score of `0.503 ± 0.076`


## We use the same train, valid, test dataset as their model, the etc data comes from us

