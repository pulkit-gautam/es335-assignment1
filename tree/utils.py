"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import math


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """

    if y.dtype == int or y.dtype == float:
        return True
    else:
        return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """

    value_counts = Y.value_counts()

    total_instances = len(Y)
    entropy = 0
    for count in value_counts:
        probability = count / total_instances
        entropy -= probability * math.log2(probability)

    return entropy
    


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """

    value_counts = Y.value_counts()

    total_instances = len(Y)
    gini_index = 1
    for count in value_counts:
        probability = count / total_instances
        gini_index -= probability**2
    
    return gini_index


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """

    entropy_original = entropy(Y)

    weighted_entropy = 0
    unique_values = attr.unique()
    total_instances = len(Y)

    for value in unique_values:
        subset_Y = Y[attr == value]
        subset_size = len(subset_Y)
        weight = subset_size / total_instances
        subset_entropy = entropy(subset_Y)
        weighted_entropy += weight * subset_entropy

    information_gain = entropy_original - weighted_entropy

    return information_gain

    


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

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

    pass
