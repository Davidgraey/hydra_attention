import numpy as np
import matplotlib.pyplot as plt

loss_dictionary = {}

loss_func = lambda f: loss_dictionary.setdefault(f.__name__, f)
EPSILON = 1e-10


derivative_dictionary = {}

derivative = lambda f: derivative_dictionary.setdefault(f.__name__, f)

x_axis = np.linspace(-1, 1, 100)

@loss_func
def rmse(prediction, T, pointwise = True):
    '''
    ROOT MEAN SQUARED ERRORS FOR REGRESSION MODELS
    :param prediction: numpy array of predictions (outputs)
    :param T: True or T values - dimensionally match prediction
    :return: evaluation of error / loss
    '''
    return np.expand_dims(np.sqrt(np.mean((T - prediction) ** 2)), -1)


@loss_func
def sse(prediction, T, pointwise = True):
    '''
    SUM OF SQUARED ERRORS FOR REGRESSION MODELS
    :param prediction: numpy array of predictions (outputs)
    :param T: True or T values - dimensionally match prediction
    :return: evaluation of error / loss
    '''
    return np.expand_dims(np.sum((T - prediction)**2), -1)


@loss_func
def mse(prediction, T, pointwise = True):
    '''
    MEAN SQUARED ERRORS (L2) FOR REGRESSION MODELS
    :param prediction: numpy array of predictions (outputs)
    :param T: True or T values - dimensionally match prediction
    :return: evaluation of error / loss
    '''
    loss = np.expand_dims(0.5 * np.mean((prediction - T) ** 2), -1)
    #print(loss)
    return loss


@loss_func
def mae(prediction, T, pointwise = True):
    '''
    MEAN ABSOLUTE ERROR (L1) FOR REGRESSION MODELS
    :param prediction: numpy array of predictions (outputs)
    :param T: True or T values - dimensionally match prediction
    :return: evaluation of error / loss
    '''
    return np.expand_dims(np.sum(np.abs(T - prediction)), -1)


@loss_func
def logcosh(prediction, T, pointwise = True):
    '''
    LOG COSH FOR REGRESSION MODELS
    :param prediction: numpy array of predictions (outputs)
    :param T: True or T values - dimensionally match prediction
    :return: evaluation of error / loss
    '''
    return np.expand_dims(np.sum(np.log(np.cosh(T - prediction))), -1)


@loss_func
def hinge(prediction, T, pointwise = True):
    '''
    HINGE LOSS FOR CLASSIFICATION MODELS
    :param prediction: numpy array of predictions (outputs)
    :param T: True or T values - dimensionally match prediction
    :return: evaluation of error / loss
    '''
    return np.expand_dims(np.max(0, 1 - prediction * T), -1)


@loss_func
def cross_entropy(prediction, T, pointwise = True):
    '''
    CROSS ENTROPY LOSS FOR SINGLE LABEL, MULTICLASS CLASSIFICATION - Softmax
    Assumes that T is one-hot encoded vector [0, 0, 1, 0]
    :param prediction: numpy array of predictions (outputs)
    :param T: True or T values - dimensionally match prediction
    :return: evaluation of error / loss
    '''
    #loss = np.sum(T * np.log(prediction + EPSILON) + (1 - T) * np.log((1 - prediction) + EPSILON))
    #loss = -np.sum(T * np.log(prediction + EPSILON))
    #loss = np.sum(T + np.log(prediction))
    #np.sum(np.nan_to_num(-T * np.log(prediction) - (1 - T) * np.log(1 - prediction)))
    #(-1/T.shape[0]) * np.sum(np.nan_to_num(-T * np.log(prediction) - (1 - T) * np.log(1 - prediction)))
    # loss = np.nan_to_num(np.sum((-T * np.log(prediction) - (1 - T) * np.log(1 - prediction))))
    loss = np.nan_to_num(-np.mean((T * np.log(prediction) + (1 - T) * np.log(1 - prediction))))

    #print(f'loss is {loss}')
    return loss


@loss_func
def multi_label_cross_entropy(prediction, T):
    '''
    CROSS ENTROPY LOSS FOR MULTILABEL, MULTICLASS CLASSIFICATION - SIG
    Assumes that T is one-hot encoded vector [0, 0, 1, 0]
    :param prediction: numpy array of predictions (outputs)
    :param T: True or T values - dimensionally match prediction
    :return: evaluation of error / loss
    '''
    n_classes = T.shape[-1]
    loss = (-1/n_classes) * np.sum(T *np.log(prediction) + (1-T * np.log(1-prediction)),axis = 1)
    print(loss.shape)
    return loss

#-------
@loss_func
def cwd(prediction, T):
    '''
    Context Window Difference - Summed difference of prediction vs T across word window
    :param prediction: numpy array of predictions (outputs) - representing a single word in sequence
    :param T: T in this case represents the context window around our prediction word
    :return:
    '''
    return np.sum([np.subtract(prediction, word) for word in T], axis = 0)

#------------------------------------------------------------------
#DERIVATIVES

@derivative
def mse_derivative(prediction, T):
    return prediction - T

@derivative
def mae_derivative(prediction, T):
    return (prediction - T) / np.abs(prediction - T)

@derivative
def rmse_derivative(prediction, T):
    return np.abs(T - prediction) / np.sqrt(prediction.shape[0])

@derivative
def cross_entropy_derivative(prediction, T):
    #return np.sum(T / (prediction + EPSILON))
    #return -np.log(prediction) - np.log(1-prediction)
    # 2ln(y) OR 2Y - 1 / x
    #return np.nan_to_num(1 - T) / ((1 - prediction) - (T / prediction) + EPSILON)
    #return np.nan_to_num(2*T - (1 / prediction + EPSILON))
    return prediction - T

@derivative
def multi_label_cross_entropy_derivative(prediction, T):
    n_classes = T.shape[-1]
    return (prediction - T) / (n_classes * (prediction - (prediction * prediction)))
    #return (T - prediction) / (prediction - (prediction * prediction)))
    #return (prediction - T) / (prediction(1-prediction))
    #return prediction - T