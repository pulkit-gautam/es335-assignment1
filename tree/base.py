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

class Node():
    def __init__(self):
        self.discrete=False        
        self.split_val=None         
        self.value=None             
        self.child={}                        
        self.isleaf=False
        self.attribute=None

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to
    root = None

    def __init__(self, criterion, max_depth=5):
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth

    def fit_tree(self,x,y,depth):
      node=Node()
      max_entropy_cat = -1*float("inf")
      max_mse_real = float("inf")
      best_attribute = -1
      splitval = None
      y1 = y.to_numpy()
      # classification if y is discrete
      if not check_ifreal(y):

        unique_classes = y.unique()
        if unique_classes.size == 1: # only one value for prediction
          node.isleaf = True #predict directly
          node.discrete = True # for discrete attribute
          node.value = np.random.choice(unique_classes)
          return node
        
        if self.max_depth==depth or x.shape[1]==0: #if max depth is reached
          node.isleaf = True
          node.discrete = True
          node.value = np.bincount(y1).argmax()
          return node
        
        for index in x:
          att = x[index]
          if(att.dtype == "category"): # checking for discrete attributes
            if(self.criterion=="information_gain"):
              
              info_gain = information_gain(y,att,"information_gain")
              
            else:
              info_gain = 0
              att=list(att)
              length=len(att)
              lab={}
              for k in range(len(att)):
                if att[k] in lab.keys():
                  lab[att[k]].append(y1[k])
                else:
                  lab[att[k]]=[y1[k]]

              for val in lab.values():
                info_gain -= (len(val)/length)*gini_index(pd.Series(val))
            if(info_gain>max_entropy_cat):
                max_entropy_cat = info_gain
                best_attribute = index

          else: # if real attributes
            att = att.sort_values(ascending=True)
            for j in range(att.shape[0]-1):
              info_gain = None
              split = (att[j]+att[j+1])/2
              left = pd.Series([y1[k] for k in range(y1.size) if att[k]<=split])
              right = pd.Series([y1[k] for k in range(y1.size) if att[k]>split])
              if(self.criterion=="information_gain"):
                
                initial_entropy = entropy(y)
                
                
                left_entropy = entropy(left)
                right_entropy = entropy(right)
                gain= initial_entropy - (left_entropy * (left.size / len(y1)) + right_entropy * (right.size / len(y1)))
              else:
                info_gain = (-1/len(y1))*((left.size*gini_index(left) + right.size*gini_index(right)))
              if(info_gain != None):
                if(info_gain > max_entropy_cat):
                  max_entropy_cat = info_gain
                  best_attribute = index
                  splitval = split
              else:
                  gain= max_entropy_cat + 10000
                  best_attribute = index
                  splitval=split
      else: #means regression
        if(self.max_depth==depth or y1.size==1 or x.shape[1]==0):
          node.isleaf=True
          node.value=y1.mean()
          return node
        
        for index in x:
          att=x[index]
          if(att.dtype=="category"): # checking for discrete attributes
              unique_vals = np.unique(att)
              info_gain = 0
              for j in unique_vals:
                  y_sub = pd.Series([y[k] for k in range(y.size) if att[k] == j])
                  info_gain += y_sub.size*(variance(y_sub))
                  if(max_mse_real > info_gain):
                            max_mse_real = info_gain
                            best_attribute = index
                            splitval = None

          else: # real input
            att = att.sort_values(ascending=True)
            for j in range(y.shape[0]-1):
              split = (att[j]+att[j+1])/2
              left=[]
              right=[]
              
              left = ([y[k] for k in range(y.size) if att[k]<=split])
              left = np.asarray(left)
              
              right = ([y[k] for k in range(y.size) if att[k]>split])
              right = np.asarray(right)
              mse = np.sum(np.square(np.mean(left)-left)) + np.sum(np.square(np.mean(right)-right))
              if(mse < max_mse_real):
                max_mse_real = mse
                best_attribute = index
                splitval = split
      if(splitval==None):
        node.discrete = True
        node.attribute=best_attribute
        
        classes = np.unique(x[best_attribute])
      
        for j in classes:
          y_modify = pd.Series([y1[k] for k in range(y1.size) if x[best_attribute][k]==j], dtype=y1.dtype)
          x_modify = x[x[best_attribute]==j].reset_index().drop(['index',best_attribute],axis=1) 
          node.child[j] = self.fit_tree(x_modify, y_modify, depth+1)

        
      else:
        node.attribute = best_attribute
        node.split_val = splitval
        val_left = []
        val_right = []
        x_left = x[x[best_attribute]<=splitval].reset_index().drop(['index'],axis=1)
        x_right = x[x[best_attribute]>splitval].reset_index().drop(['index'],axis=1)
        for j in range(len(x[best_attribute])):
          if x[best_attribute][j]<=splitval:
            val_left.append(y1[j])

          else:
            val_right.append(y1[j])
        val_left=pd.Series(val_left)  
        
        val_right=pd.Series(val_right)
          
        node.child["left"]=self.fit_tree(x_left, val_left, depth+1)
        node.child["right"]=self.fit_tree(x_right, val_right, depth+1)
      return node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        self.root = self.fit_tree(X,y,0)

    '''
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
    '''

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        output=list()
        for i in range(X.shape[0]):
          x = X.iloc[i,:] #selecting ith row for prediction
          head = self.root
          while(not head.isleaf):
            if(head.discrete == True) :# discrete attribute
              head = head.child[x[head.attribute]]
            else:                     #real att
              if(x[head.attribute] <= head.split_val):
                head = head.child["left"]
              else:
                head = head.child["right"]  
          output.append(head.value)
        return pd.Series(output) 

    '''
    def predict_tree(self, node, row):
            # Base case: If the node is a leaf, return the result
            if node.leaf is not None:
                return node.leaf

            # Decide which subtree to traverse based on the split condition
            if row[node.attribute] <= node.value:
                return self.predict_tree(node.left, row)
            else:
                return self.predict_tree(node.right, row)
    '''
    

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
        head=self.root
        self.plot_tree(head,0)

    def plot_tree(self,head,depth):
        if head.isleaf:
            print("prediction: ",head.value)
        else:
            if(head.discrete) : # if attribute is discrete
                for i in head.child.keys() :
                    print(f"?{head.attribute} == {i}")
                    print("\t"*(depth+1),end="")
                    self.plot_tree(head.child[i],depth+1)
                    print("\t"*depth,end="")
            else: #real attribte
                print(f"?(X[{head.attribute}] > {head.split_val})")
                print("\t"*(depth+1),"Y:= ",end="")
                self.plot_tree(head.child["left"],depth+1)
                print("\t"*(depth+1),"N:= ",end="")  
                self.plot_tree(head.child["right"],depth+1) 

