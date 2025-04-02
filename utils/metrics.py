import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


from sklearn import metrics

def evaluate(true, pred):
    true_mean = np.mean(true)
    pred = np.nan_to_num(pred,nan=true_mean,posinf=true_mean,neginf=true_mean)
    mae = metrics.mean_absolute_error(true,pred)
    mse = metrics.mean_squared_error(true,pred)
    rmse = metrics.root_mean_squared_error(true,pred)
    r2 = metrics.r2_score(true,pred)
    # print('MAE:',mae, 'MSE:', mse,'RMSE:',rmse,'R2 Square:',r2)
    return {'MAE':mae, 'MSE': mse,'RMSE':rmse,'R2':r2}#mae, mse, rmse,r2