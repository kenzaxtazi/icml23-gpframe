import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def mll(y_true, y_pred, v_pred) -> float:
    """
    Mean Log Loss (MLL)

    Args:
        y_true : targets
        y_pred : posterior means
        v_pred : posterior variances

    Returns:
        int: Mean Log Loss
    """
    # set everything to numpy arrays
    y_true, y_pred, v_pred = np.array(
        y_true), np.array(y_pred), np.array(v_pred)
    std_pred = np.sqrt(v_pred)
    first_term = 0.5 * np.log(2 * np.pi * std_pred**2)
    second_term = ((y_true - y_pred)**2)/(2 * std_pred**2)
    return np.mean(first_term + second_term)


def RMSE95(y_true: np.array, y_pred: np.array) -> float:
    """
    RMSE of 95th percentile

    Args:
        y_true (np.array): targets
        y_pred (np.array): posterior means

    Returns:
        _type_: _description_
    """
    p95 = np.percentile(y_true, 95.0)
    indx = [y_true >= p95][0]
    y_true_p95 = y_true[indx]
    y_pred_p95 = y_pred[indx]
    rmse95 = mean_squared_error(
        y_true_p95, y_pred_p95, squared=False)
    return rmse95


def RMSE5(y_true, y_pred) -> float:
    """
    RMSE of 5th percentile

    Args:
        y_true (np.array): targets
        y_pred (np.array): posterior means

    Returns:
        _type_: _description_
    """
    p5 = np.percentile(y_true, 5.0)
    indx = [y_true <= p5][0]
    y_true_p5 = y_true[indx]
    y_pred_p5 = y_pred[indx]
    rmse5 = mean_squared_error(y_true_p5, y_pred_p5, squared=False)
    return rmse5
