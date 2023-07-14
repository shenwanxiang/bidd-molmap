# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 17:10:53 2020

@author: wanxiang.shen@u.nus.edu
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import get_scorer, SCORERS
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .cbks import CLA_EarlyStoppingAndPerformance, Reg_EarlyStoppingAndPerformance
from .net import MolMapNet, MolMapDualPathNet, MolMapAddPathNet, MolMapResNet
from .loss import cross_entropy, weighted_cross_entropy


from joblib import dump, load
from  copy import copy
from tensorflow.keras.models import load_model as load_tf_model

import gc
import tensorflow.keras.backend as K
from numba import cuda

def save_model(model, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print('saving model to %s' % model_path)
    model_new = copy(model)
    model_new._model.save(os.path.join(model_path, 'inner_model.h5'))
    model_new._model = None
    model_new._performance = None
    res = dump(model_new,  os.path.join(model_path, 'outer_model.est'))
    return res


    
def load_model(model_path, gpuid=None):
    '''
    gpuid: load model to specific gpu: {None, 0, 1, 2, 3,..}
    '''
    model = load(os.path.join(model_path, 'outer_model.est'))
    if gpuid==None:
        gpuid = model.gpuid
    else:
        gpuid = str(gpuid)
    os.environ["CUDA_VISIBLE_DEVICES"]= gpuid
    model.gpuid = gpuid
    model._model = load_tf_model(os.path.join(model_path, 'inner_model.h5'))
    return model


def clean(clf): 
    del clf._model
    del clf._performance
    del clf
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph() # TF graph isn't same as Keras graph
 
    
class RegressionEstimator(BaseEstimator, RegressorMixin):
    
    """ An MolMap CNN Regression estimator 
    Parameters
    ----------
    n_outputs: int,
        the number of the outputs, in case it is a multi-task
    fmap_shape1: tuple
        width, height, and channels of the first input feature map
    fmap_shape2: tuple, default = None
        width and height of the second input feature map
    epochs : int, default = 100
        A parameter used for training epochs. 
    dense_layers: list, default = [128]
        A parameter used for the dense layers.    
    monitor: str
        {'val_loss', 'val_r2'}
        
    
    Examples
    --------

    """
    
    def __init__(self, 
                 n_outputs,
                 fmap_shape1,
                 fmap_shape2 = None,
                 epochs = 800,  
                 conv1_kernel_size = 13,
                 dense_layers = [128, 64],  
                 dense_avf = 'relu',
                 batch_size = 128,  
                 lr = 1e-4, 
                 loss = 'mse',
                 monitor = 'val_loss', 
                 metric = 'r2',
                 patience = 50,
                 verbose = 2, 
                 random_state = 32,
                 y_scale = None, #None, minmax, standard
                 name = "Regression Estimator",
                 gpuid = "0",
                ):
        
        self.n_outputs = n_outputs
        self.fmap_shape1 = fmap_shape1
        self.fmap_shape2 = fmap_shape2
        
        self.epochs = epochs
        self.dense_layers = dense_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.dense_avf = dense_avf
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.monitor = monitor
        self.metric = metric
        self.patience = patience
        
        
        self.verbose = verbose
        self.random_state = random_state
        assert y_scale in [None, 'minmax', 'standard'], "scale_y should be None, or 'minmax', or 'standard'!"
        if y_scale == None:
            y_scaler = None
        elif y_scale == 'minmax':
            y_scaler = MinMaxScaler()
        elif y_scale == 'standard':
            y_scaler = StandardScaler()
        
        self.y_scaler = y_scaler
        self.y_scale = y_scale
        
        self.name = name

        self.gpuid = str(gpuid)
        os.environ["CUDA_VISIBLE_DEVICES"]= self.gpuid
        
        np.random.seed(self.random_state)
        tf.compat.v1.set_random_seed(self.random_state)
        
        if self.fmap_shape2 is None:
            model = MolMapNet(self.fmap_shape1,
                              n_outputs = self.n_outputs, 
                              conv1_kernel_size = self.conv1_kernel_size,
                              dense_layers = self.dense_layers, 
                              dense_avf = self.dense_avf, 
                              last_avf = 'linear')

        else:
            model = MolMapDualPathNet(self.fmap_shape1,
                                      self.fmap_shape2,
                                      n_outputs = self.n_outputs, 
                                      conv1_kernel_size = self.conv1_kernel_size,
                                      dense_layers = self.dense_layers, 
                                      dense_avf = self.dense_avf, 
                                      last_avf = 'linear')
        
        opt = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
        model.compile(optimizer = opt, loss = self.loss)
        
        self._model = model
        
        print(self)
        
        
    def get_params(self, deep=True):
 
        model_paras =  {
                        "n_outputs": self.n_outputs,
                        "fmap_shape1": self.fmap_shape1,
                        "fmap_shape2": self.fmap_shape2,
                        "epochs": self.epochs, 
                        "lr":self.lr, 
                        "loss":self.loss, 
                        "conv1_kernel_size": self.conv1_kernel_size,
                        "dense_layers": self.dense_layers, 
                        "dense_avf":self.dense_avf, 
                        "batch_size":self.batch_size, 
                        "monitor": self.monitor,
                        "patience":self.patience,
                        "random_state":self.random_state,
                        "verbose":self.verbose,
                        "y_scale": self.y_scale,
                        "name":self.name,
                        "gpuid":self.gpuid,
                       }
        return model_paras
    

    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

        
    def count_model_params(self, deep=True):
        model_paras = self._model.count_params()
        return model_paras
    
    
    def fit(self, X, y,  X_valid = None, y_valid = None):

        # Check that X and y have correct shape

        if self.fmap_shape2 is None:
            if  X.ndim != 4:
                raise ValueError("Found array X with dim %d. %s expected == 4." % (X.ndim, self.name))
            w, h, c = X.shape[1:]
            w_, h_, c_ = self.fmap_shape1
            assert (w == w_) & (h == h_) & (c == c_), "Input shape of X is not matched the defined fmap_shape. expected == %s" % self.fmap_shape1
        
        else:
            if len(X) != 2:
                raise ValueError("Input X should be a tuple with two elements." )
            X1, X2 = X
            w1_, h1_, c1_ = self.fmap_shape1
            w2_, h2_, c2_ = self.fmap_shape2
            w1, h1, c1 = X1.shape[1:]
            w2, h2, c2 = X2.shape[1:]
            assert (w1 == w1_) & (h1 == h1_) & (c1 == c1_), "Input shape of X is not matched the defined fmap_shape. expected == %s" % self.fmap_shape1
            assert (w2 == w2_) & (h2 == h2_) & (c2 == c2_), "Input shape of X is not matched the defined fmap_shape. expected == %s" % self.fmap_shape2

            
        self.X_ = X
        self.y_ = y
        if (X_valid is None) | (y_valid is None):
            X_valid = X
            y_valid = y
        
        if self.y_scaler != None:
            self.y_scaler = self.y_scaler.fit(y)
            y = self.y_scaler.transform(y)
            if y_valid is not None:
                y_valid = self.y_scaler.transform(y_valid)
            
        performance = Reg_EarlyStoppingAndPerformance((X, y), 
                                                      (X_valid, y_valid), 
                                                      y_scaler = self.y_scaler,
                                                      patience = self.patience, 
                                                      criteria = self.monitor,
                                                      verbose = self.verbose,
                                                      batch_size = self.batch_size)

        history = self._model.fit(X, y, 
                                  batch_size=self.batch_size, 
                                  epochs= self.epochs, verbose= 0, shuffle = True, 
                                  validation_data = (X_valid, y_valid), 
                                  callbacks=[performance]) 

        self._performance = performance
        self.history = self._performance.history

        return self


    
    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_w, n_features_h, n_features_c)
            Vector to be scored, where `n_samples` is the number of samples and

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        
        # Check is fit had been called
        check_is_fitted(self)
        
        y_pred = self._model.predict(X, batch_size=self.batch_size)
        
        if self.y_scaler != None:
            y_pred = self.y_scaler.inverse_transform(y_pred)
            
        return y_pred
    


    def score(self, X, y, scoring = 'r2'):
        """Returns the score using the `scoring` option on the given
        test data and labels.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        scoring: str, default: r2, 
            {'r2', 'rmse'}
        
        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """
 
        rmse_list, r2_list = self._performance.evaluate(X, y)
        
        if scoring == 'r2':
            myscore = np.nanmean(r2_list)
        else:
            myscore = np.nanmean(rmse_list)
            
        return myscore
    
    
    def clean(self):
        clean(self)    
    
    
    def save_model(self, model_path):
        return save_model(self, model_path)

    
    def load_model(self, model_path, gpuid=None):
        return load_model(model_path, gpuid=gpuid)
    
    
    
    
class MultiClassEstimator(BaseEstimator, ClassifierMixin):

    """ An MolMap CNN MultiClass estimator
    Parameters
    ----------
    epochs : int, default = 150
        A parameter used for training epochs. 
    dense_layers: list, default = [128]
        A parameter used for the dense layers.    
    
    Examples
    --------


    """
    
    def __init__(self, 
                 n_outputs,
                 fmap_shape1,
                 fmap_shape2 = None,
                 
                 epochs = 800,  
                 conv1_kernel_size = 13,
                 dense_layers = [128, 64],  
                 dense_avf = 'relu',
                 batch_size = 128,  
                 lr = 1e-4, 
                 loss = 'categorical_crossentropy',
                 monitor = 'val_loss', 
                 metric = 'ROC',
                 patience = 50,
                 verbose = 2, 
                 random_state = 32,
                 name = "MultiClass Estimator",
                 
                 gpuid = 0,
                ):
        
        self.n_outputs = n_outputs
        self.fmap_shape1 = fmap_shape1
        self.fmap_shape2 = fmap_shape2
        
        self.epochs = epochs
        self.dense_layers = dense_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.dense_avf = dense_avf
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.monitor = monitor
        self.metric = metric
        self.patience = patience
        
        
        self.verbose = verbose
        self.random_state = random_state
        
        self.name = name
        self.gpuid = str(gpuid)
        os.environ["CUDA_VISIBLE_DEVICES"]= self.gpuid
        
        np.random.seed(self.random_state)
        tf.compat.v1.set_random_seed(self.random_state)
        if self.fmap_shape2 is None:
            model = MolMapNet(self.fmap_shape1,
                              n_outputs = self.n_outputs, 
                              conv1_kernel_size = self.conv1_kernel_size,
                              dense_layers = self.dense_layers, 
                              dense_avf = self.dense_avf, 
                              last_avf = 'softmax')

        else:
            model = MolMapDualPathNet(self.fmap_shape1,
                                      self.fmap_shape2,
                                      n_outputs = self.n_outputs, 
                                      conv1_kernel_size = self.conv1_kernel_size,
                                      dense_layers = self.dense_layers, 
                                      dense_avf = self.dense_avf, 
                                      last_avf = 'softmax')
        
        opt = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
        model.compile(optimizer = opt, loss = self.loss, metrics = ['accuracy'])
        
        self._model = model
        
        print(self)
        
        
    def get_params(self, deep=True):

        model_paras =  {
                        "n_outputs": self.n_outputs,
                        "fmap_shape1": self.fmap_shape1,
                        "fmap_shape2": self.fmap_shape2,
                        "epochs": self.epochs, 
                        "lr":self.lr, 
                        "loss": self.loss,
                        "conv1_kernel_size": self.conv1_kernel_size,
                        "dense_layers": self.dense_layers, 
                        "dense_avf":self.dense_avf, 
                        "batch_size":self.batch_size, 
                        "monitor": self.monitor,
                        "metric":self.metric,
                        "patience":self.patience,
                        "random_state":self.random_state,
                        "verbose":self.verbose,
                        "name":self.name,
                        "gpuid": self.gpuid,
                       }

        return model_paras
    
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    
    def count_model_params(self, deep=True):
        model_paras = self._model.count_params()
        return model_paras
    

    def fit(self, X, y,  X_valid = None, y_valid = None):

        # Check that X and y have correct shape

        if self.fmap_shape2 is None:
            if  X.ndim != 4:
                raise ValueError("Found array X with dim %d. %s expected == 4." % (X.ndim, self.name))
            w, h, c = X.shape[1:]
            w_, h_, c_ = self.fmap_shape1
            assert (w == w_) & (h == h_) & (c == c_), "Input shape of X is not matched the defined fmap_shape. expected == %s" % self.fmap_shape1
        
        else:
            if len(X) != 2:
                raise ValueError("Input X should be a tuple with two elements." )
            X1, X2 = X
            w1_, h1_, c1_ = self.fmap_shape1
            w2_, h2_, c2_ = self.fmap_shape2
            w1, h1, c1 = X1.shape[1:]
            w2, h2, c2 = X2.shape[1:]
            assert (w1 == w1_) & (h1 == h1_) & (c1 == c1_), "Input shape of X is not matched the defined fmap_shape. expected == %s" % self.fmap_shape1
            assert (w2 == w2_) & (h2 == h2_) & (c2 == c2_), "Input shape of X is not matched the defined fmap_shape. expected == %s" % self.fmap_shape2
            
        self.X_ = X
        self.y_ = y
        if (X_valid is None) | (y_valid is None):
            X_valid = X
            y_valid = y
        
        performance = CLA_EarlyStoppingAndPerformance((X, y), (X_valid, y_valid), 
                                                      patience = self.patience, 
                                                      criteria = self.monitor,
                                                      metric = self.metric,  
                                                      last_avf="softmax",
                                                      verbose = self.verbose, 
                                                      batch_size = self.batch_size)

        history = self._model.fit(X, y, 
                                  batch_size=self.batch_size, 
                                  epochs= self.epochs, verbose= 0, shuffle = True, 
                                  validation_data = (X_valid, y_valid), 
                                  callbacks=[performance]) 

        self._performance = performance
        self.history = self._performance.history
        
        # Return the classifier
        return self



    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        # Check is fit had been called
        check_is_fitted(self)
        y_prob = self._model.predict(X, batch_size=self.batch_size)
        return y_prob
    
    
    
    
    def predict(self, X):
        
        # Check is fit had been called
        check_is_fitted(self)
        probs = self.predict_proba(X)
        y_pred = pd.get_dummies(np.argmax(probs, axis=1)).values
        return y_pred


    

    def score(self, X, y):
        """Returns the accuracy score of metric used in init
        test data and labels.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """
        
        metrics = self._performance.evaluate(X, y)
        return np.nanmean(metrics)
    
    
    def clean(self):
        clean(self)    
    
    
    def save_model(self, model_path):
        return save_model(self, model_path)

    
    def load_model(self, model_path, gpuid=None):
        return load_model(model_path, gpuid=gpuid)    
    
    
    
    

class MultiLabelEstimator(BaseEstimator, ClassifierMixin):

    """ An MolMAP CNN MultiLabel estimator
    Parameters
    ---------- 
    
    Examples
    --------

    """
    
    def __init__(self, 
                 
                 n_outputs,
                 fmap_shape1,
                 fmap_shape2 = None,
                 
                 epochs = 800,  
                 conv1_kernel_size = 13,
                 dense_layers = [128, 64],  
                 dense_avf = 'relu',
                 batch_size = 128,  
                 lr = 1e-4, 
                 loss = cross_entropy,
                 monitor = 'val_loss', 
                 metric = 'ROC',
                 patience = 50,
                 verbose = 2, 
                 random_state = 32,
                 name = "MultiLabels Estimator",
                 gpuid = 0,
                ):
        
        self.n_outputs = n_outputs
        self.fmap_shape1 = fmap_shape1
        self.fmap_shape2 = fmap_shape2
        
        self.epochs = epochs
        self.dense_layers = dense_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.dense_avf = dense_avf
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.monitor = monitor
        self.metric = metric
        self.patience = patience
        
        
        self.verbose = verbose
        self.random_state = random_state
        
        self.name = name
        self.gpuid = str(gpuid)
        os.environ["CUDA_VISIBLE_DEVICES"]= self.gpuid        
        
        np.random.seed(self.random_state)
        tf.compat.v1.set_random_seed(self.random_state)
        if self.fmap_shape2 is None:
            model = MolMapNet(self.fmap_shape1,
                              n_outputs = self.n_outputs, 
                              conv1_kernel_size = self.conv1_kernel_size,
                              dense_layers = self.dense_layers, 
                              dense_avf = self.dense_avf, 
                              last_avf = None)

        else:
            model = MolMapDualPathNet(self.fmap_shape1,
                                      self.fmap_shape2,
                                      n_outputs = self.n_outputs, 
                                      conv1_kernel_size = self.conv1_kernel_size,
                                      dense_layers = self.dense_layers, 
                                      dense_avf = self.dense_avf, 
                                      last_avf = None)
        
        
        opt = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
        model.compile(optimizer = opt, loss = self.loss)
        
        self._model = model
        print(self)
        
        
    def get_params(self, deep=True):

        model_paras =  {
                        "n_outputs": self.n_outputs,
                        "fmap_shape1": self.fmap_shape1,
                        "fmap_shape2": self.fmap_shape2,
            
                        "epochs": self.epochs, 
                        "lr":self.lr, 
                        "loss":self.loss,
                        "conv1_kernel_size": self.conv1_kernel_size,
                        "dense_layers": self.dense_layers, 
                        "dense_avf":self.dense_avf, 
                        "batch_size":self.batch_size, 
                        "monitor": self.monitor,
                        "metric":self.metric,
                        "patience":self.patience,
                        "random_state":self.random_state,
                        "verbose":self.verbose,
                        "name":self.name,
                        "gpuid": self.gpuid,
                       }

        return model_paras
    
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    
    def count_model_params(self, deep=True):
        model_paras = self._model.count_params()
        return model_paras
    

    def fit(self, X, y,  X_valid = None, y_valid = None):

        # Check that X and y have correct shape
        if self.fmap_shape2 is None:
            if  X.ndim != 4:
                raise ValueError("Found array X with dim %d. %s expected == 4." % (X.ndim, self.name))
            w, h, c = X.shape[1:]
            w_, h_, c_ = self.fmap_shape1
            assert (w == w_) & (h == h_) & (c == c_), "Input shape of X is not matched the defined fmap_shape. expected == %s" % self.fmap_shape1
        
        else:
            if len(X) != 2:
                raise ValueError("Input X should be a tuple with two elements." )
            X1, X2 = X
            w1_, h1_, c1_ = self.fmap_shape1
            w2_, h2_, c2_ = self.fmap_shape2
            w1, h1, c1 = X1.shape[1:]
            w2, h2, c2 = X2.shape[1:]
            assert (w1 == w1_) & (h1 == h1_) & (c1 == c1_), "Input shape of X is not matched the defined fmap_shape. expected == %s" % self.fmap_shape1
            assert (w2 == w2_) & (h2 == h2_) & (c2 == c2_), "Input shape of X is not matched the defined fmap_shape. expected == %s" % self.fmap_shape2
            
        self.X_ = X
        self.y_ = y
        if (X_valid is None) | (y_valid is None):
            X_valid = X
            y_valid = y

        performance = CLA_EarlyStoppingAndPerformance((X, y), 
                                                      (X_valid, y_valid), 
                                                      patience = self.patience, 
                                                      criteria = self.monitor,
                                                      metric = self.metric,  
                                                      last_avf=None,
                                                      verbose = self.verbose,batch_size = self.batch_size)

        history = self._model.fit(X, y, 
                                  batch_size=self.batch_size, 
                                  epochs= self.epochs, verbose= 0, shuffle = True, 
                                  validation_data = (X_valid, y_valid), 
                                  callbacks=[performance]) 

        self._performance = performance  
        self.history = self._performance.history
        # Return the classifier
        return self



    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        # Check is fit had been called
        check_is_fitted(self)
        y_prob = self._model.predict(X, batch_size=self.batch_size)
        y_prob = self._performance.sigmoid(y_prob)
        return y_prob
    
    
    
    
    def predict(self, X):
        
        # Check is fit had been called
        check_is_fitted(self)
        y_pred = np.round(self.predict_proba(X))
        return y_pred
    
    

    def score(self, X, y):
        """Returns the accuracy score of metric used in init
        test data and labels.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """
        
        metrics = self._performance.evaluate(X, y)
        return np.nanmean(metrics)
    

    def evaluate(self, X, y):
        """Returns the accuracy score of metric used in init
        test data and labels.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        metrics : float
            Score of self.predict(X) wrt. y.
        """
        
        metrics = self._performance.evaluate(X, y)
        return metrics
    
    
    def clean(self):
        clean(self)    
    
    
    def save_model(self, model_path):
        return save_model(self, model_path)

    
    def load_model(self, model_path, gpuid=None):
        return load_model(model_path, gpuid=gpuid)