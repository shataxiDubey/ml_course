from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    return (y_hat == y).sum() / len(y)


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    y_hat_true = y_hat[y_hat==cls]
    y_actual = y[y_hat_true.index]
    y_precision = y_hat_true == y_actual
    if len(y_hat_true) == 0:
        return 0
    return (y_precision == True).sum()/len(y_hat_true)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    y_true = y[y == cls]
    y_pred_true = y_hat[y_true.index]
    y_recall = y_pred_true == y_true
    return (y_recall == True).sum()/len(y_true)


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    return (((y_hat - y)**2).mean())**0.5


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    return abs((y_hat - y)).mean()
