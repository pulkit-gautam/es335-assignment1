"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np
from scipy.special import xlogy
from sklearn.model_selection import train_test_split

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if y.dtype == int or y.dtype == float:
        return True
    
    else:
        return False
    pass


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    prob = Y.value_counts(normalize=True)
    return -np.sum(prob * np.log2(prob))
    pass


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    count = Y.value_counts()
    prob = count/len(Y)

    gini_index = 1 - sum((p**2) for p in prob)
    return gini_index
    pass


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    original_entropy = entropy(Y)

    weighted_sum_entropy = 0
    unique_values = attr.unique()

    for value in unique_values:
        subset_indices = attr[attr == value].index
        subset_entropy = entropy(Y.loc[subset_indices])
        weighted_sum_entropy += (len(subset_indices) / len(attr)) * subset_entropy

    info_gain = original_entropy - weighted_sum_entropy

    return info_gain
    pass

def mse_calculate(Y: pd.Series) -> float:
    mean_y = Y.mean()
    MSE = ((Y - mean_y) ** 2).mean()
    return MSE

def find_best_split_mse(X: pd.DataFrame, y: pd.Series, feature):
    unique_values = X[feature].unique()
    best_split_point = None
    best_mse = float('inf')

    for value in unique_values:
        left_indices = X[feature] <= value
        right_indices = ~left_indices

        left_mse = mse_calculate(y[left_indices])
        right_mse = mse_calculate(y[right_indices])

        total_mse = left_mse + right_mse

        if total_mse < best_mse:
            best_mse = total_mse
            best_split_point = value

    return best_split_point

def find_best_split_cat(X: pd.DataFrame, y: pd.Series, feature):
    unique_values = X[feature].unique()
    best_split_point = None
    best_gini = float('inf')

    for value in unique_values:
        left_indices = X[feature] == value
        right_indices = ~left_indices

        left_gini = gini_index(y[left_indices])
        right_gini = gini_index(y[right_indices])

        total_gini = ((len(y[left_indices]) / len(y))) * left_gini + ((len(y[right_indices]) / len(y))) * right_gini

        if total_gini < best_gini:
            best_gini = total_gini
            best_split_point = value

    return best_split_point


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    opt_attribute = None
    best_score = float('inf') 

    if check_ifreal(X[attribute]):
        for attribute in features:
            split_point = find_best_split_mse(X, y, attribute)
            mse = mse_calculate(y[X[attribute] <= split_point]) + mse_calculate(y[X[attribute] > split_point])

            if mse < best_score:
                best_score = mse
                opt_attribute = attribute

    else:
        for attribute in features:
            split_point = find_best_split_cat(X, y, attribute)
            gini = gini_index(y[X[attribute] <= split_point]) + gini_index(y[X[attribute] > split_point])

            if gini < best_score:
                best_score = gini
                opt_attribute = attribute

    return opt_attribute
    pass
    


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    if check_ifreal(X[attribute]):
        mask = X[attribute] <= value
    else:
        mask = X[attribute] == value

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=mask)

    return X_train, X_test, y_train, y_test
    pass
