from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import os
import numpy as np

from scipy.stats.stats import pearsonr
def r2_score(x,y):
    pcc, _ = pearsonr(x,y)
    return pcc**2
  
    
class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the validate data loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=5, criteria = 'val_loss' ):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.criteria = criteria

        
    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf            
        self.best_epoch = 0
        
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.criteria)
        

        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('\nRestoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)


        
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('\nEpoch %05d: early stopping' % (self.stopped_epoch + 1))

            

            


class ClassificationPerformance(tf.keras.callbacks.Callback):

    
    def __init__(self, train_data, valid_data, MASK = -1):
        super(ClassificationPerformance, self).__init__()
        
        self.x, self.y  = train_data
        self.x_val, self.y_val = valid_data
        
        self.history = {'training_loss':[],
                        'validation_loss':[],
                        
                        'training_auc':[],
                        'validation_auc':[],
                        
                        'epoch':[]}
        self.MASK = MASK
        
    def sigmoid(self, x):
        s = 1/(1+np.exp(-x))
        return s

    
    def roc_auc(self, y_true, y_pred):

        y_pred_logits = self.sigmoid(y_pred)
        N_classes = y_pred_logits.shape[1]

        aucs = []
        for i in range(N_classes):
            y_pred_one_class = y_pred_logits[:,i]
            y_true_one_class = y_true[:, i]
            mask = ~(y_true_one_class == self.MASK)
            auc = roc_auc_score(y_true_one_class[mask], y_pred_one_class[mask])
            aucs.append(auc)
        return aucs    

        
        
    def on_epoch_end(self, epoch, logs={}):
        
        y_pred = self.model.predict(self.x)
        roc_list = self.roc_auc(self.y, y_pred)
        roc_mean = np.nanmean(roc_list)
        
        y_pred_val = self.model.predict(self.x_val)
        roc_val_list = self.roc_auc(self.y_val, y_pred_val)        
        roc_val_mean = np.nanmean(roc_val_list)
        
        self.history['training_loss'].append(logs.get('loss'))
        self.history['validation_loss'].append(logs.get('val_loss'))
        self.history['training_auc'].append(roc_mean)
        self.history['validation_auc'].append(roc_val_mean)
        self.history['epoch'].append(epoch)
        
        
        eph = str(epoch+1).zfill(4)        
        loss = '{0:.4f}'.format((logs.get('loss')))
        val_loss = '{0:.4f}'.format((logs.get('val_loss')))
        auc = '{0:.4f}'.format(roc_mean)
        auc_val = '{0:.4f}'.format(roc_val_mean)

                                    
        print('\repoch: %s, loss: %s - val_loss: %s; auc: %s - auc_val: %s' % (eph,
                                                                               loss, 
                                                                               val_loss, 
                                                                               auc,
                                                                               auc_val), end=100*' '+'\n')

    def evaluate(self, testX, testY):
        
        y_pred = self.model.predict(testX)
        roc_list = self.roc_auc(testY, y_pred)
        return roc_list



    



class RegressionPerformance(tf.keras.callbacks.Callback):

    
    def __init__(self, train_data, valid_data, MASK = np.nan):
        super(RegressionPerformance, self).__init__()
        
        self.x, self.y  = train_data
        self.x_val, self.y_val = valid_data
        
        self.history = {'training_loss':[],
                        'validation_loss':[],
                        
                        'training_rmse':[],
                        'validation_rmse':[],                        
                        'training_r2':[],
                        'validation_r2':[],
                        
                        'epoch':[]}
        self.MASK = MASK

        
    def rmse(self, y_true, y_pred):

        N_classes = y_pred.shape[1]
        rmses = []
        for i in range(N_classes):
            y_pred_one_class = y_pred[:,i]
            y_true_one_class = y_true[:, i]
            mask = ~(y_true_one_class == self.MASK)
            mse = mean_squared_error(y_true_one_class[mask], y_pred_one_class[mask])
            rmse = np.sqrt(mse)
            rmses.append(rmse)
        return rmses   
    
    
    def r2(self, y_true, y_pred):
        N_classes = y_pred.shape[1]
        r2s = []
        for i in range(N_classes):
            y_pred_one_class = y_pred[:,i]
            y_true_one_class = y_true[:, i]
            mask = ~(y_true_one_class == self.MASK)
            r2 = r2_score(y_true_one_class[mask], y_pred_one_class[mask])
            r2s.append(r2)
        return r2s   
    
        
    def on_epoch_end(self, epoch, logs={}):
        
        y_pred = self.model.predict(self.x)
        rmse_list = self.rmse(self.y, y_pred)
        rmse_mean = np.nanmean(rmse_list)
        
        r2_list = self.r2(self.y, y_pred) 
        r2_mean = np.nanmean(r2_list)
        
        
        y_pred_val = self.model.predict(self.x_val)
        rmse_list_val = self.rmse(self.y_val, y_pred_val)        
        rmse_mean_val = np.nanmean(rmse_list_val)
        
        r2_list_val = self.r2(self.y_val, y_pred_val)       
        r2_mean_val = np.nanmean(r2_list_val)        
        
        self.history['training_loss'].append(logs.get('loss'))
        self.history['validation_loss'].append(logs.get('val_loss'))
        
        self.history['training_rmse'].append(rmse_mean)
        self.history['validation_rmse'].append(rmse_mean_val)
        
        self.history['training_r2'].append(r2_mean)
        self.history['validation_r2'].append(r2_mean_val)        
        
        self.history['epoch'].append(epoch)

        # logs is a dictionary
        eph = str(epoch+1).zfill(4)   
        loss = '{0:.4f}'.format((logs.get('loss')))
        val_loss = '{0:.4f}'.format((logs.get('val_loss')))
        rmse = '{0:.4f}'.format(rmse_mean)
        rmse_val = '{0:.4f}'.format(rmse_mean_val)
        r2_mean = '{0:.4f}'.format(r2_mean)
        r2_mean_val = '{0:.4f}'.format(r2_mean_val)
        
        print('\repoch: %s, loss: %s - val_loss: %s; rmse: %s - rmse_val: %s;  r2: %s - r2_val: %s' % (eph,
                                                                                                       loss, val_loss, 
                                                                                                       rmse,rmse_val,
                                                                                                       r2_mean,r2_mean_val),
              end=100*' '+'\n')
        

    def evaluate(self, testX, testY):
        """evalulate, return rmse and r2"""
        y_pred = self.model.predict(testX)
        rmse_list = self.rmse(testY, y_pred)
        r2_list = self.r2(testY, y_pred)
        return rmse_list, r2_list

    

    

    

