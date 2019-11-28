import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import roc_auc_score

class EarlyStoppingAndPerformance(tf.keras.callbacks.Callback):

    def __init__(self, train_data, valid_data, MASK = -1, patience=5, criteria = 'val_loss'):
        super(EarlyStoppingAndPerformance, self).__init__()
        
        assert criteria in ['val_loss', 'val_auc'], 'not support %s ! only %s' % (criteria, ['val_loss', 'val_auc'])
        self.x, self.y  = train_data
        self.x_val, self.y_val = valid_data
        
        self.history = {'loss':[],
                        'val_loss':[],
                        'auc':[],
                        'val_auc':[],
                        
                        'epoch':[]}
        self.MASK = MASK
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.criteria = criteria
        

        
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
            try:
                auc = roc_auc_score(y_true_one_class[mask], y_pred_one_class[mask])
            except:
                auc = np.nan
            aucs.append(auc)
        return aucs  
    
        
        
    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        if self.criteria == 'val_loss':
            self.best = np.Inf  
        else:
            self.best = -np.Inf
            
        self.best_epoch = 0
        
 
        
    def on_epoch_end(self, epoch, logs={}):
        
        y_pred = self.model.predict(self.x)
        roc_list = self.roc_auc(self.y, y_pred)
        roc_mean = np.nanmean(roc_list)
        
        y_pred_val = self.model.predict(self.x_val)
        roc_val_list = self.roc_auc(self.y_val, y_pred_val)        
        roc_val_mean = np.nanmean(roc_val_list)
        
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['auc'].append(roc_mean)
        self.history['val_auc'].append(roc_val_mean)
        self.history['epoch'].append(epoch)
        
        
        eph = str(epoch+1).zfill(4)        
        loss = '{0:.4f}'.format((logs.get('loss')))
        val_loss = '{0:.4f}'.format((logs.get('val_loss')))
        auc = '{0:.4f}'.format(roc_mean)
        auc_val = '{0:.4f}'.format(roc_val_mean)                 
        print('\repoch: %s, loss: %s - val_loss: %s; auc: %s - val_auc: %s' % (eph,
                                                                               loss, 
                                                                               val_loss, 
                                                                               auc,
                                                                               auc_val), end=100*' '+'\n')


        if self.criteria == 'val_loss':
            current = logs.get(self.criteria)
            if current <= self.best:
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
                    
        else:
            current = roc_val_mean
            if current >= self.best:
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

        
    def evaluate(self, testX, testY):
        
        y_pred = self.model.predict(testX)
        roc_list = self.roc_auc(testY, y_pred)
        return roc_list            

            
