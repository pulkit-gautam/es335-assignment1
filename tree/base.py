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
      #Defining the values and parameters
      node = Node()
      max_entropy_cat = -1*float("inf")
      max_mse_real = float("inf")
      best_attribute = -1
      splitval = None
      y1 = y.to_numpy()

      # if y is discrete
      if not check_ifreal(y):
        unique_classes = y.unique()
        # only one value for prediction
        if unique_classes.size == 1: 
          node.isleaf = True 
          node.discrete = True 
          node.value = np.random.choice(unique_classes)
          return node
        #if max depth is reached
        if self.max_depth == depth or x.shape[1] == 0: 
          node.isleaf = True
          node.discrete = True
          node.value = np.bincount(y1).argmax()
          return node
        
        for index in x:
          att = x[index]
          # checking for discrete attributes
          #Discrete input and discrete output
          if(att.dtype == "category"): 
            #Checking for criterion
            if(self.criterion=="information_gain"):
              info_gain = information_gain(y,att,"entropy")
            else:
              info_gain = information_gain(y,att,"gini_index")
              
            if(info_gain > max_entropy_cat):
                max_entropy_cat = info_gain
                best_attribute = index

          #Real input and discrete output
          else: 
            
              #Calculating info gain based on criterion
              if(self.criterion=="information_gain"):
                info_gain,split = information_gain(y,att,"entropy")
              else:
                info_gain,split = information_gain(y,att,"gini index")
              #if info gain succesfully calculated
              if(info_gain != None):
                #Update the info gain and best attribute to split on
                if(info_gain > max_entropy_cat):
                  max_entropy_cat = info_gain
                  best_attribute = index
                  splitval = split
              #If info gain not calculated, update its value to some high number
              else:
                  info_gain= max_entropy_cat + 10000
                  best_attribute = index
                  splitval=split

      #means real output i.e. regression
      else: 
        #Checking for max depth and size of data
        if(self.max_depth==depth or y1.size==1 or x.shape[1]==0):
          node.isleaf=True
          node.value=y1.mean()
          return node
        
        for index in x:
          att=x[index]
          # checking for discrete attributes
          #Discrete input and Real Output
          if(att.dtype=="category"): 
              info_gain = information_gain(y,att,"entropy")
              if(max_mse_real > info_gain):
                            max_mse_real = info_gain
                            best_attribute = index
                            splitval = None

          #Real input and Real Output
          else: 
            mse,split = information_gain(y,att,"entropy")
            if(mse < max_mse_real):
                max_mse_real = mse
                best_attribute = index
                splitval = split

      #If no split value         
      if(splitval==None):
        node.discrete = True
        node.attribute=best_attribute
        
        classes = np.unique(x[best_attribute])

        #making new x and y which are a subset of original if they have values same as that of best attribute
        for j in classes:
          y_modify = pd.Series([y1[k] for k in range(y1.size) if x[best_attribute][k]==j], dtype=y1.dtype)
          x_modify = x[x[best_attribute]==j].reset_index().drop(['index',best_attribute],axis=1) 
          #recursively calling the function for the child node with the updated x and y
          node.child[j] = self.fit_tree(x_modify, y_modify, depth+1)
      
      else:
        node.attribute = best_attribute
        node.split_val = splitval
        val_left = []
        val_right = []
        #Splitting into left and right subsets based on split value
        x_left = x[x[best_attribute]<=splitval].reset_index().drop(['index'],axis=1)
        x_right = x[x[best_attribute]>splitval].reset_index().drop(['index'],axis=1)
        for j in range(len(x[best_attribute])):
          if x[best_attribute][j]<=splitval:
            val_left.append(y1[j])
          else:
            val_right.append(y1[j])

        val_left=pd.Series(val_left)
        val_right=pd.Series(val_right)
        
        #Recursively calling the function on the left and right child subset 
        node.child["left"]=self.fit_tree(x_left, val_left, depth+1)
        node.child["right"]=self.fit_tree(x_right, val_right, depth+1)

      return node
    '''
    def fit_tree(self,x,y,depth):
      #Defining the values and parameters
      node = Node()
      max_entropy_cat = -1*float("inf")
      max_mse_real = float("inf")
      best_attribute = -1
      splitval = None
      y1 = y.to_numpy()

      # if y is discrete
      if not check_ifreal(y):
        unique_classes = y.unique()
        # only one value for prediction
        if unique_classes.size == 1: 
          node.isleaf = True 
          node.discrete = True 
          node.value = np.random.choice(unique_classes)
          return node
        #if max depth is reached
        if self.max_depth == depth or x.shape[1] == 0: 
          node.isleaf = True
          node.discrete = True
          node.value = np.bincount(y1).argmax()
          return node
        
        for index in x:
          att = x[index]
          # checking for discrete attributes
          #Discrete input and discrete output
          if(att.dtype == "category"): 
            #Checking for criterion
            if(self.criterion=="information_gain"):
              info_gain = information_gain(y,att,"information_gain")
            else:
              info_gain = 0
              att = list(att)
              length = len(att)
              lab = {}
              for k in range(length):
                #if the attribute is already a key in the dictionary
                if att[k] in lab.keys():
                  lab[att[k]].append(y1[k])
                else:
                  lab[att[k]]=[y1[k]]
              #Calculating the gini index
              for val in lab.values():
                info_gain -= (len(val)/length)*gini_index(pd.Series(val))
            #Updating the information gain and best attribute
            if(info_gain > max_entropy_cat):
                max_entropy_cat = info_gain
                best_attribute = index

          #Real input and discrete output
          else: 
            att = att.sort_values(ascending=True)
            for j in range(att.shape[0]-1):
              info_gain = None
              #Assuming a split point between 2 values of an attribute and splitting into 2 subsets left and right  
              split = (att[j]+att[j+1])/2
              left = pd.Series([y1[k] for k in range(y1.size) if att[k]<=split])
              right = pd.Series([y1[k] for k in range(y1.size) if att[k]>split])
              #Calculating info gain based on criterion
              if(self.criterion=="information_gain"):
                initial_entropy = entropy(y)
                left_entropy = entropy(left)
                right_entropy = entropy(right)
                gain= initial_entropy - (left_entropy * (left.size / len(y1)) + right_entropy * (right.size / len(y1)))
              else:
                info_gain = (-1/len(y1))*((left.size*gini_index(left) + right.size*gini_index(right)))
              #if info gain succesfully calculated
              if(info_gain != None):
                #Update the info gain and best attribute to split on
                if(info_gain > max_entropy_cat):
                  max_entropy_cat = info_gain
                  best_attribute = index
                  splitval = split
              #If info gain not calculated, update its value to some high number
              else:
                  gain= max_entropy_cat + 10000
                  best_attribute = index
                  splitval=split

      #means real output i.e. regression
      else: 
        #Checking for max depth and size of data
        if(self.max_depth==depth or y1.size==1 or x.shape[1]==0):
          node.isleaf=True
          node.value=y1.mean()
          return node
        
        for index in x:
          att=x[index]
          # checking for discrete attributes
          #Discrete input and Real Output
          if(att.dtype=="category"): 
              unique_vals = np.unique(att)
              info_gain = 0
              for j in unique_vals:
                  #subset containing the data which satisfies the condition
                  y_sub = pd.Series([y[k] for k in range(y.size) if att[k] == j])
                  info_gain += y_sub.size*(variance(y_sub))
                  #Updating the mse and best attribute to split upon
                  if(max_mse_real > info_gain):
                            max_mse_real = info_gain
                            best_attribute = index
                            splitval = None

          #Real input and Real Output
          else: 
            att = att.sort_values(ascending=True)
            for j in range(y.shape[0]-1):
              split = (att[j]+att[j+1])/2
              left=[]
              right=[]
              #inserting data points into left and right subset by comparing them to split point
              left = ([y[k] for k in range(y.size) if att[k]<=split])
              left = np.asarray(left)
              
              right = ([y[k] for k in range(y.size) if att[k]>split])
              right = np.asarray(right)
              #Calculating the mse of both subsets
              mse = np.sum(np.square(np.mean(left)-left)) + np.sum(np.square(np.mean(right)-right))
              #Updating the mse and best attribute to split upon
              if(mse < max_mse_real):
                max_mse_real = mse
                best_attribute = index
                splitval = split

      #If no split value         
      if(splitval==None):
        node.discrete = True
        node.attribute=best_attribute
        
        classes = np.unique(x[best_attribute])

        #making new x and y which are a subset of original if they have values same as that of best attribute
        for j in classes:
          y_modify = pd.Series([y1[k] for k in range(y1.size) if x[best_attribute][k]==j], dtype=y1.dtype)
          x_modify = x[x[best_attribute]==j].reset_index().drop(['index',best_attribute],axis=1) 
          #recursively calling the function for the child node with the updated x and y
          node.child[j] = self.fit_tree(x_modify, y_modify, depth+1)
      
      else:
        node.attribute = best_attribute
        node.split_val = splitval
        val_left = []
        val_right = []
        #Splitting into left and right subsets based on split value
        x_left = x[x[best_attribute]<=splitval].reset_index().drop(['index'],axis=1)
        x_right = x[x[best_attribute]>splitval].reset_index().drop(['index'],axis=1)
        for j in range(len(x[best_attribute])):
          if x[best_attribute][j]<=splitval:
            val_left.append(y1[j])
          else:
            val_right.append(y1[j])

        val_left=pd.Series(val_left)
        val_right=pd.Series(val_right)
        
        #Recursively calling the function on the left and right child subset 
        node.child["left"]=self.fit_tree(x_left, val_left, depth+1)
        node.child["right"]=self.fit_tree(x_right, val_right, depth+1)

      return node
    '''

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        self.root = self.fit_tree(X,y,0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        #Traverse the constructed tree for predicting the values of the inputs
        output = list()
        for i in range(X.shape[0]):
          #selecting all the columns of the ith row for prediction
          x = X.iloc[i,:] 
          head = self.root
          while(not head.isleaf):
            #discrete attribute
            if(head.discrete == True):
              #Traversing into the correct child node by comparing the values
              head = head.child[x[head.attribute]]
            #real attribute
            else:
              #comparing the value of head attribute of the current row and using that to traverse into the correct child                     
              if(x[head.attribute] <= head.split_val):
                head = head.child["left"]
              else:
                head = head.child["right"]

          output.append(head.value)

        return pd.Series(output) 
    
    def plot_tree(self,head,depth):
        #checking if the node is a leaf node or not
        if head.isleaf:
            print("prediction: ",head.value)

        else:
            #discrete attribute
            if(head.discrete) : 
                for i in head.child.keys() :
                    print(f"?{head.attribute} == {i}")
                    print("\t"*(depth+1),end="")
                    #recursively plotting for each child
                    self.plot_tree(head.child[i],depth+1)
                    print("\t"*depth,end="")
            #real attribute
            else: 
                print(f"?(X[{head.attribute}] > {head.split_val})")
                #recursively plotting for left and right child
                print("\t"*(depth+1),"Y:= ",end="")
                self.plot_tree(head.child["left"],depth+1)
                print("\t"*(depth+1),"N:= ",end="")  
                self.plot_tree(head.child["right"],depth+1) 

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
      
        self.plot_tree(self.root,0)

    

