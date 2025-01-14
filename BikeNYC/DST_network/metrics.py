# import numpy as np
from tensorflow.python.keras import backend as K

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def root_mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


# aliases
mse = MSE = mean_squared_error
# rmse = RMSE = root_mean_square_error


def masked_mean_squared_error(y_true, y_pred):
    idx = (y_true > 1e-6).nonzero()
    return K.mean(K.square(y_pred[idx] - y_true[idx]))

def masked_rmse(y_true, y_pred):
    return masked_mean_squared_error(y_true, y_pred) ** 0.5


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

	
threshold=0.05

def mean_absolute_percentage_error(y_true, y_pred):
    return K.mean( K.abs(y_pred-y_true) / K.maximum(K.cast(threshold,'float32'),y_true+1.0) )

def mape(y_true, y_pred):
    return K.mean( K.abs(y_pred-y_true) / K.maximum(K.cast(threshold,'float32'),y_true+1.0) )

