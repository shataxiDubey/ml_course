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
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

# @dataclass
# class DTree:
#     condition: dict()  # List of conditions to be satisfied to reach the node
#     CHILDREN: dict()
#     Label: Union[int, float, str]    # Label contains the output class. 
#     Name: str = None
#     condition_type: str = None

#     def isleafNode(self):
#         if len(self.condition) == 0:
#             return self.Label
#         return None
    
#     def addCondition(self, cond, NODE):
#         self.condition.update(cond)
#         self.CHILDREN.update(NODE)  
        

#     def getChild(self, value):
#         for i in range(len(self.CHILDREN)):
#             if self.condition[i] == value:
#                 return self.CHILDREN[i]
#         return None    
    
#     def addLabel(self, label):
#         self.Label = label

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int = 100 # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

    # def build_decision_tree(self, X, Y, depth = 0):
    #     if len(Y.unique()) == 1:
    #         return DTree({}, {}, Y.unique()[0])
    #     elif len(X.columns) == 0:
    #         return DTree({}, {}, Y.mode()[0])
    #     elif self.max_depth == depth:
    #         return DTree({}, {}, Y.mode()[0])
    #     else:
    #         bestAttr, split_value = self.findBestAttr(X, Y)
    #         tree = DTree({}, {}, None, bestAttr)
    #         if X[bestAttr].dtype != 'category':
    #             X_subset_le = X[X[bestAttr] <= split_value]
    #             Y_subset_le = Y[X[bestAttr] <= split_value]
    #             X_subset_gt = X[X[bestAttr] > split_value]
    #             Y_subset_gt = Y[X[bestAttr] > split_value]
    #             subtree1 = self.build_decision_tree(X_subset_le, Y_subset_le, depth+1)
    #             cond1 = { (split_value,'<='): Condition('<=', split_value)}
    #             node = {(split_value,'<='): subtree1}
    #             tree.addCondition(cond1, node)
    #             tree.condition_type = 'real'
    #             subtree2 = self.build_decision_tree(X_subset_gt, Y_subset_gt, depth+1)
    #             cond2 = {(split_value,'>') : Condition('>', split_value)}
    #             node = {(split_value,'>'): subtree2}
    #             tree.addCondition(cond2, node)
    #             tree.condition_type = 'real'
    #         else:
    #             for value in X[bestAttr].unique():
    #                 X_subset = X[X[bestAttr] == value].drop(bestAttr, axis=1)
    #                 Y_subset = Y[X[bestAttr] == value]

    #                 subtree = self.build_decision_tree(X_subset, Y_subset, depth+1)
    #                 cond = {(value,'='): Condition('==', value)}
    #                 node = {(value, '='): subtree}
    #                 tree.addCondition(cond, node)
    #                 tree.condition_type = 'categorical'
    #     return tree
        pass
         

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        X_clone = X.copy()
        Y_clone = y.copy()
        decision_tree = self.build_decision_tree(X_clone, Y_clone)
        self.root = decision_tree
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        pass

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
        pass
