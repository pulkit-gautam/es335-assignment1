"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np
from scipy.special import xlogy


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    #If the values are real return True
    if y.dtype == int or y.dtype == float:
        return True
    
    else:
        return False
    

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    #Calculating the entrpy using the xlogy method
    prob = Y.value_counts(normalize=True)
    return -np.sum(prob * np.log2(prob))
    

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    #Finding the total count of unique values
    count = Y.value_counts()
    #Finding the prob of a value to occur
    prob = count/len(Y)
    #Calculating the gini index
    gini_index = 1 - sum((p**2) for p in prob)
    return gini_index
    

def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    #Finding the original entropy
    original_entropy = entropy(Y)
    #Initializing the weighted entropy sum and finding the unique values in the Series
    weighted_sum_entropy = 0
    unique_values = attr.unique()

    for value in unique_values:
        #Finding indices of entries which have the attribute value of the loop 
        subset_indices = attr[attr == value].index
        #Finding the entropy of this series
        subset_entropy = entropy(Y.loc[subset_indices])
        #Weighted entropy of the above series
        weighted_sum_entropy += (len(subset_indices) / len(attr)) * subset_entropy

    #Calculating the information gain
    info_gain = original_entropy - weighted_sum_entropy

    return info_gain
    

def mse_calculate(Y: pd.Series) -> float:
    #Calculating the MSE
    mean_y = Y.mean()
    MSE = ((Y - mean_y) ** 2).mean()
    return MSE


def find_best_split_mse(X: pd.DataFrame, y: pd.Series, feature):
    #Finding unique values 
    unique_values = X[feature].unique()
    best_split_point = None
    best_mse = float('inf')

    for value in unique_values:
        #Spitting the entries based on the current value which we are finding the best split for
        left_indices = X[feature] <= value
        right_indices = ~left_indices
        #Calculating the MSE of the left and the right splits
        left_mse = mse_calculate(y[left_indices])
        right_mse = mse_calculate(y[right_indices])

        total_mse = ((len(y[left_indices]) / len(y))) * left_mse + ((len(y[right_indices]) / len(y))) * right_mse
        #Updating the best MSE and split point
        if total_mse < best_mse:
            best_mse = total_mse
            best_split_point = value

    return best_split_point


def find_best_split_cat(X: pd.DataFrame, y: pd.Series, feature):
    #Finding unique values
    unique_values = X[feature].unique()
    best_split_point = None
    best_gini = float('inf')

    for value in unique_values:
        #Spitting the entries based on the current value which we are finding the best split for
        left_indices = X[feature] == value
        right_indices = ~left_indices
        #Calculating the gini of the left and the right splits
        left_gini = gini_index(y[left_indices])
        right_gini = gini_index(y[right_indices])

        total_gini = ((len(y[left_indices]) / len(y))) * left_gini + ((len(y[right_indices]) / len(y))) * right_gini
        #Updating the best gini and split point
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

    #Initializing the Optimal Attribute and the Score. In case of Real input, the Score is lowest MSE and in case of Discrete input, the score is lowest gini
    opt_attribute = None
    best_score = float('inf') 

    #Real Valued Input
    if check_ifreal(X[attribute]):
        for attribute in features:
            #Finding the best split in a particular attribute
            split_point = find_best_split_mse(X, y, attribute)
            #Calculating the MSE of that split
            mse = mse_calculate(y[X[attribute] <= split_point]) + mse_calculate(y[X[attribute] > split_point])
            #Updating the best score and the optimal attribute
            if mse < best_score:
                best_score = mse
                opt_attribute = attribute

    #Discrete Valued Input
    else:
        if criterion == 'gini_index':
            for attribute in features:
                #Finding the best split in a particular attribute
                split_point = find_best_split_cat(X, y, attribute)
                #Calculating the gini of that split
                gini = gini_index(y[X[attribute] <= split_point]) + gini_index(y[X[attribute] > split_point])
                #Updating the best score and the optimal attribute
                if gini < best_score:
                    best_score = gini
                    opt_attribute = attribute
        
        else:
            #Initializing information gain and best attribute
            info_gain = 0.0
            best_attribute = None
            #Picking out the best feature based on maximum information gain
            for attribute in features:
                curr_info_gain = information_gain(y,X[attribute])
                if curr_info_gain > info_gain:
                    info_gain = curr_info_gain
                    best_attribute = attribute
            return best_attribute
                

    return opt_attribute


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

    #Numerical Attribute
    if check_ifreal(X[attribute]):  
        mask = X[attribute] <= value
        
    #Categorical Attribute
    else:  
        mask = X[attribute] == value

    #Spliting the data based on mask
    X_left = X[mask]
    y_left = y[mask]
    X_right = X[~mask]
    y_right = y[~mask]

    return X_left, y_left, X_right, y_right
    