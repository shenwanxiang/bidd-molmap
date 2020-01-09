# In this Section, we compare our molmap model with the graph-based model: AttentiveFP from [Xiong et al's work] (https://pubs.acs.org/doi/abs/10.1021/acs.jmedchem.9b00959) 


The split train, valid and test datasets are taken from [AttentiveFP](https://github.com/OpenDrugAI/AttentiveFP/tree/master/code) code repo directly

-----


we dump their split result by running their codes, for example, for the ESOL dataset, we run the https://github.com/OpenDrugAI/AttentiveFP/blob/master/code/2_Physical_Chemistry_ESOL.ipynb code, at the line of 7 by the following script:

```python
from joblib import load, dump
dump((train_df, valid_df, test_df), './01_ESOL_attentiveFP.data')

```
Note that, for malaria dataset, they only split dataset into train, test subsets by a fraction of 0.2 using 68 as the random_seed :
``` python
random_seed = 68
test_df = remained_df.sample(frac=0.2,random_state=random_seed)
train_df = remained_df.drop(test_df.index)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

from joblib import load, dump
dump((train_df, test_df), '04_Malaria_attentiveFP.data')
```


## The ToxCast and MUV data sets are not included because of the splitting bugs in their codes: the training set contains parts of data set that in test and valid data sets.