# 1.Install chemprop first

https://github.com/chemprop/chemprop

```bash
cd /path/to/chemprop
conda env create -f environment.yml
source activate chemprop
```


# 2. Performance Test

## optimize HPs

```bash
python hyperparameter_optimization.py --data_path DMPNN/train_dmpnn.csv --dataset_type regression --num_iters 3 --config_save_path DMPNN/logs_config
```

## training and test model by optimal params.

```bash
python train.py --data_path DMPNN/train_dmpnn.csv --dataset_type regression --config_path DMPNN/logs_config --save_dir DMPNN/logs_checkpoints1 --separate_val_path DMPNN/valid_dmpnn.csv
```

```bash
# test etc dataset
python train.py --data_path DMPNN/train_dmpnn.csv --dataset_type regression --config_path DMPNN/logs_config --save_dir DMPNN/logs_checkpoints2 --separate_val_path DMPNN/valid_dmpnn.csv
```

```bash
python train.py --data_path DMPNN/train_dmpnn.csv --dataset_type regression --config_path DMPNN/logs_config --save_dir DMPNN/logs_checkpoints3 --separate_val_path DMPNN/valid_dmpnn.csv
```


## predict

```bash
python predict.py --test_path DMPNN/train_dmpnn.csv --checkpoint_dir DMPNN/logs_checkpoints1 --preds_path DMPNN/pred_train_dmpnn_1.csv
python predict.py --test_path DMPNN/valid_dmpnn.csv --checkpoint_dir DMPNN/logs_checkpoints1 --preds_path DMPNN/pred_valid_dmpnn_1.csv
python predict.py --test_path DMPNN/test_dmpnn.csv --checkpoint_dir DMPNN/logs_checkpoints1 --preds_path DMPNN/pred_test_dmpnn_1.csv
python predict.py --test_path DMPNN/etc_dmpnn.csv --checkpoint_dir DMPNN/logs_checkpoints1 --preds_path DMPNN/pred_etc_dmpnn_1.csv
```

```bash
python predict.py --test_path DMPNN/train_dmpnn.csv --checkpoint_dir DMPNN/logs_checkpoints2 --preds_path DMPNN/pred_train_dmpnn_2.csv
python predict.py --test_path DMPNN/valid_dmpnn.csv --checkpoint_dir DMPNN/logs_checkpoints2 --preds_path DMPNN/pred_valid_dmpnn_2.csv
python predict.py --test_path DMPNN/test_dmpnn.csv --checkpoint_dir DMPNN/logs_checkpoints2 --preds_path DMPNN/pred_test_dmpnn_2.csv
python predict.py --test_path DMPNN/etc_dmpnn.csv --checkpoint_dir DMPNN/logs_checkpoints2 --preds_path DMPNN/pred_etc_dmpnn_2.csv
```

```bash
python predict.py --test_path DMPNN/train_dmpnn.csv --checkpoint_dir DMPNN/logs_checkpoints3 --preds_path DMPNN/pred_train_dmpnn_3.csv
python predict.py --test_path DMPNN/valid_dmpnn.csv --checkpoint_dir DMPNN/logs_checkpoints3 --preds_path DMPNN/pred_valid_dmpnn_3.csv
python predict.py --test_path DMPNN/test_dmpnn.csv --checkpoint_dir DMPNN/logs_checkpoints3 --preds_path DMPNN/pred_test_dmpnn_3.csv
python predict.py --test_path DMPNN/etc_dmpnn.csv --checkpoint_dir DMPNN/logs_checkpoints3 --preds_path DMPNN/pred_etc_dmpnn_3.csv
```




# 3. Speed test

``` python
import time, os
import pandas as pd

start = time.time()
res = []
for epoch in [1, 10, 50, 100, 150, 300, 500]:
    
    cmd = '/home/sxh/anaconda3/envs/chemprop/bin/python train.py --data_path DMPNN/train_dmpnn.csv --dataset_type regression --config_path DMPNN/logs_config --save_dir DMPNN/logs_speed --separate_val_path DMPNN/valid_dmpnn.csv --batch_size 200 --epochs %s' % epoch

    status = os.system(cmd)
    
    end = time.time()
    total = end - start
    
    res.append([epoch, total])
    
x = pd.DataFrame(res, columns = ['epochs', 'total_time(s)'])
x['average_time(s)'] =  x['total_time(s)'] / x['epochs']
x.to_csv('./speed_DMPNN.csv')

```