import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np




###########  regression ##############
def loss_rmse(y_true, y_pred):
    '''
    Root Mean Squared Error Loss
    '''
    with tf.name_scope('rmse'):    
        res = tf.losses.mean_squared_error(y_true, y_pred)
        return tf.sqrt(res)


def metrics_r_squared(y_true, y_pred):
    '''
    R_squared computes the coefficient of determination.
    It is a measure of how well the observed outcomes are replicated by the model.
    '''
    with tf.name_scope('r_squared'):    
        residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
        total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
        r2 = tf.subtract(1.0, tf.compat.v1.div(residual, total))
        return r2

    

###########  classification ##############
def cross_entropy(y_true, y_pred, MASK = -1):
    mask = tf.cast(tf.not_equal(y_true, MASK), tf.keras.backend.floatx())
    labels = y_true * mask
    logits = y_pred * mask
    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return cost





def weighted_binary_cross_entropy_without_softmax(y_true, y_pred, weight=1.) :
    #https://stackoverflow.com/questions/43390162/class-weights-in-binary-classification-model-with-keras
    
    #y_true = tf.clip(y_true, K.epsilon(), 1)
    #y_pred = tf.clip(y_pred, K.epsilon(), 1)
    
    # fiter out the NaNs, y_true may has NaN values
    mask = tf.logical_not(tf.math.is_nan(y_true))[:,0]
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    
    logloss = -(y_true * tf.log(y_pred) * weight + (1 - y_true) * tf.log(1 - y_pred))
    res = tf.reduce_mean(logloss)

    return res






def weighted_cross_entropy(y_true, y_pred, pos_weight, MASK = -1):
    
    mask = tf.cast(tf.not_equal(y_true, MASK), tf.keras.backend.floatx())
    labels = y_true * mask
    logits = y_pred * mask    
    cost = tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, 
                                                    pos_weight = pos_weight)
    return cost

    

    



