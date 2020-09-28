# data description

----
# 1) molnet dataset
## bace.csv: the orignal bace dataset with pIC50 value, downloaded from molnet: http://moleculenet.ai/datasets-1
## the train.csv, valid.csv, test.csv set are generated from bace.csv by scaffold split method using chemprop split data tools(seed:2): https://github.com/chemprop/chemprop/blob/master/scripts/split_data.py

----
# 2) chembl dataset

## bace_chembl: dataset that comes from ChEMBL database, it has excluded the compounds in bace.csv 
## bace_chembl_novel: dataset that comes from ChEMBL database, it scaffolds is qiute different from bace.csv, it is novel structures
## bace_chembl_common: dataset that comes from ChEMBL database, it is the dataset of bace_chembl.csv that exclude bace_chembl_novel.csv
* bace_chembl = bace_chembl_novel + bace_chembl_common



----
# 3) other dataset
## drug.csv: clinical drugs that targting BACE, taken from literatures, https://www.nature.com/articles/nrd.2017.43/tables/1
## scaffold_*.csv, inhibitor, non-inhibitor.csv: scaffold is picked via tree of the BACE: http://bidd.group/molmap/BACE/BACE.html



