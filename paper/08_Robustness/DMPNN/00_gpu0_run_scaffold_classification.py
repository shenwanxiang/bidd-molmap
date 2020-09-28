import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] ="0"

# task_names = ["FreeSolv", "ESOL", "Lipop", "Malaria", "HIV", "BACE", "BBBP"]
# task_types = ["regression", "regression", "regression", "regression", "classification", "classification", "classification"]

# task_names = [ "ESOL" ]
# task_types = ["regression"]

pythoner = "/home/sxh/anaconda3/envs/chemprop/bin/python"
optimizer = "/home/sxh/Research/chemprop/hyperparameter_optimization.py"
trainer = "/home/sxh/Research/chemprop/train.py"
predicter = "/home/sxh/Research/chemprop/predict.py"
file_path = "/raid/shenwanxiang/08_Robustness/dataset_induces/split"


task_names = ["BACE" , "BBBP", "HIV" ]
task_types = ["classification", "classification", "classification"]

random_seeds = [2, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


for task_name,task_type  in zip(task_names, task_types):

    for seed in random_seeds: #

        print(task_name, seed)

        tr_path = os.path.join(file_path, task_name,"%s" % seed, "train.csv")
        
        vl_path = os.path.join(file_path, task_name,"%s" % seed, "val.csv")
        vl_pred_path = os.path.join(file_path, task_name,"%s" % seed,"DMPNN_pred_valid.csv")

        ts_path = os.path.join(file_path, task_name,"%s" % seed,"test.csv")
        ts_pred_path = os.path.join(file_path, task_name,"%s" % seed, "DMPNN_pred_test.csv")


        if (os.path.exists(vl_pred_path)) & (os.path.exists(ts_pred_path)):
            continue            

        cf_path = os.path.join(file_path, task_name,"%s" % seed, "config")
        model_save_path = os.path.join(file_path, task_name,"%s" % seed, "model_checkpoints")

        #--num_iters 5 
        cmd_opt_params = "%s %s --data_path %s --dataset_type %s --num_iters 3 --config_save_path %s" % (pythoner, optimizer, 
                                                                                                          tr_path, task_type,
                                                                                                          cf_path)
        
        if task_name == 'HIV':
            cmd_train = "%s %s --data_path %s --dataset_type %s --config_path %s --save_dir %s --separate_val_path %s" % (pythoner,  
                                                                                                                      trainer, 
                                                                                                                      tr_path, 
                                                                                                                      task_type, 
                                                                                                                      cf_path, 
                                                                                                                      model_save_path, 
                                                                                                                      vl_path)
            
        else:
            cmd_train = "%s %s --data_path %s --dataset_type %s --config_path %s --epochs 200 --save_dir %s --separate_val_path %s" % (pythoner,  
                                                                                                                      trainer, 
                                                                                                                      tr_path, 
                                                                                                                      task_type, 
                                                                                                                      cf_path, 
                                                                                                                      model_save_path, 
                                                                                                                      vl_path)

        cmd_predict_valid = "%s %s --test_path %s --checkpoint_dir %s --preds_path %s" % (pythoner, predicter, 
                                                                                          vl_path, model_save_path, 
                                                                                          vl_pred_path)
        cmd_predict_test  = "%s %s --test_path %s --checkpoint_dir %s --preds_path %s" % (pythoner, predicter, 
                                                                                          ts_path, model_save_path,
                                                                                          ts_pred_path)

        #tuning best parameters
        os.system(cmd_opt_params)

        #using best parameters to train, and using valid set to make an early stopping
        os.system(cmd_train)

        # make prediction
        os.system(cmd_predict_valid)
        os.system(cmd_predict_test)

