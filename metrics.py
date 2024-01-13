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

    correct_pred = (y_hat == y)

    accuracy = correct_pred.sum() / y.size
    return accuracy


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    predicted_positive = (y_hat == cls).sum()
    
    precision = true_positive / predicted_positive
    return precision
    


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    actual_positive = (y == cls).sum()
    
    recall = true_positive / actual_positive
    return recall


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    
    squared_error = (y_hat - y) ** 2
    mean_squared_error = squared_error.mean()
    rmse = mean_squared_error ** 0.5

    return rmse


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    
    absolute_error = abs(y_hat - y)
    mae = absolute_error.mean()
    
    return mae
