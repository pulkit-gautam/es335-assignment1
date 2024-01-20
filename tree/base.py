"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth

    class Node:
        def __init__(self, value=None, attribute=None, left=None, right=None, leaf=None, depth=None):
            self.value = value  # Value to compare in the decision node
            self.attribute = attribute  # Attribute to split on
            self.left = left  # Left subtree
            self.right = right  # Right subtree
            self.leaf = leaf  # Result for leaf nodes
            self.depth = depth  # Depth of the node in the tree

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        self.root = self.construct_tree(X,y,depth = 0,criterion=DecisionTree.criterion)

    def construct_tree(self, X, y, depth, criterion):
        # Base case: If all labels are the same or max depth is reached, create a leaf node
        if len(set(y)) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return self.Node(leaf=y.iloc[0], depth=depth)

        # Find the best split based on a certain criterion (e.g., Gini index, information gain)
        attributes = pd.Series(X.columns.values)
        best_split_attribute = opt_split_attribute(X, y, criterion, attributes) 
        best_info_gain, best_split_value = information_gain(X[best_split_attribute], attributes, criterion)

        # Split the dataset into left and right subsets
        left_subset = X[X[best_split_attribute] <= best_split_value]
        right_subset = X[X[best_split_attribute] > best_split_value]

        # Recursively construct the left and right subtrees
        left_subtree = self.construct_tree(left_subset, y[left_subset.index], depth + 1,criterion)
        right_subtree = self.construct_tree(right_subset, y[right_subset.index], depth + 1,criterion)

        # Create a decision node for the best split
        decision_node = self.Node(value=best_split_value,attribute=best_split_attribute, 
                                  left=left_subtree, right=right_subtree, depth=depth)

        return decision_node

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        predictions = []
        for index, row in X.iterrows():
            predictions.append(self.predict_tree(self.root, row))
        return pd.Series(predictions, index=X.index)

    def predict_tree(self, node, row):
        # Base case: If the node is a leaf, return the result
        if node.leaf is not None:
            return node.leaf

        # Decide which subtree to traverse based on the split condition
        if row[node.attribute] <= node.value:
            return self.predict_tree(node.left, row)
        else:
            return self.predict_tree(node.right, row)

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.plot_tree(self.root)

    def plot_tree(self,node,indent=""):
        if node is not None:
            # Decision node
            if node.attribute is not None:
                print(f"{indent}?(X{node.attribute} > {node.value})")
                print(f"{indent}\tY:", end=" ")
                self.plot_tree(node.left, indent + "\t\t")
                print(f"{indent}\tN:", end=" ")
                self.plot_tree(node.right, indent + "\t\t")
            # Leaf node
            else:
                print(f"{indent}Class {node.leaf}")

