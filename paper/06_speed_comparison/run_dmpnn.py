import time, os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]="6"

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
