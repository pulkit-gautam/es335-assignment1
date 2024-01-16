"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """

    if y.dtype == float or y.dtype == int:
        return True
    else:
        return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    total_samples = len(Y)
    unique_classes = Y.unique()
    entropy_value = 0

    for class_value in unique_classes:
        class_count = len(Y[Y == class_value])
        class_probability = class_count / total_samples
        entropy_value -= class_probability * np.log2(class_probability)

    return entropy_value



def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    total_samples = len(Y)
    unique_classes = Y.unique()
    gini_index_value = 1

    for class_value in unique_classes:
        class_count = len(Y[Y == class_value])
        class_probability = class_count / total_samples
        gini_index_value -= class_probability ** 2

    return gini_index_value
    

def variance(Y: pd.Series) -> float:
    """
    Function to calculate variance
    """
    if(len(Y) == 1):
        return 0
    else:
        return Y.var() 

def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """

    total_samples = len(Y)
    unique_values = attr.unique()
    
    if(check_ifreal(Y)):
        variance_before_split = variance(Y)
        information_gain_value = variance_before_split

        for value in unique_values:
            subset_Y = Y[attr == value]
            subset_size = len(subset_Y)
            subset_variance = variance(subset_Y)
            information_gain_value -= (subset_size / total_samples) * subset_variance

        return information_gain_value
    else:
        entropy_before_split = entropy(Y)
        information_gain_value = entropy_before_split

        for value in unique_values:
            subset_Y = Y[attr == value]
            subset_size = len(subset_Y)
            subset_entropy = entropy(subset_Y)
            information_gain_value -= (subset_size / total_samples) * subset_entropy

        return information_gain_value


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    best_attribute = None
    best_info_gain = -float('inf')
    best_gini_index = float('inf')

    for attribute in features:
        if criterion == 'entropy':
            info_gain = information_gain(y, X[attribute])
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_attribute = attribute
        elif criterion == 'gini':
            gini_index = gini_index(y, X[attribute])
            if gini_index < best_gini_index:
                best_gini_index = gini_index
                best_attribute = attribute

    return best_attribute


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Function to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data (Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    
    mask = X[attribute] == value
    X_split = X[mask]
    y_split = y[mask]

    return X_split, y_split
