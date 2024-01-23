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
import graphviz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *
import graphviz

pd.options.mode.chained_assignment = None

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
        self.xtype = None
        self.ytype = None

    def build_tree_dido(self, X: pd.DataFrame, y: pd.Series, depth ):

        if depth == self.max_depth or len(y.unique()) == 1 :
            # print('Y behaviour:',y,'\n')
            # print({"leaf": True, "split_value": np.bincount(y).argmax()})
            return {"leaf": True, "split_value": np.bincount(y).argmax()}
        
        best_attr = opt_split_attribute(X, y, criterion='', features= X.columns) # type: ignore

        if best_attr is None:
            # print( {"leaf": True, "split_value": y.value_counts().argmax()})
            return {"leaf": True, "split_value": y.value_counts().argmax()}

        left, right, best_val = split_data(X, y, best_attr)
        # print('left X:',left[0],'\nleft Y:',left[1],'\nright X:',right[0],'\nright y:',right[1])

        # left[0].drop([best_attr], axis = 1, inplace = True) # type: ignore
        # right[0].drop([best_attr], axis = 1, inplace = True) # type: ignore
        # print('After dropping best attr left X:',left[0],'\nleft Y:',left[1],'\nright X:',right[0],'\nright y:',right[1])

        l_st = self.build_tree_dido(left[0], left[1], depth+1) # type: ignore
        r_st = self.build_tree_dido(right[0], right[1], depth+1) # type: ignore

        # print({"leaf": False, "split_attribute": best_attr, "split_value": best_val,
        #         "left_subtree": l_st, "right_subtree": r_st})

        return {"leaf": False, "split_attribute": best_attr, "split_value": best_val,
                "left_subtree": l_st, "right_subtree": r_st}
    

    def build_tree_diro(self, X: pd.DataFrame, y: pd.Series, depth ):

        if depth == self.max_depth or ((y - y.mean())**2).mean() == 0 :
            # print('Y behaviour:',y,'\n')
            # print({"leaf": True, "split_value": y.mean()})
            return {"leaf": True, "split_value": y.mean()}
        
        best_attr = opt_split_attribute(X, y, criterion='', features= X.columns) # type: ignore

        if best_attr is None:
            # print( {"leaf": True, "split_value": y.mean()})
            return {"leaf": True, "split_value": y.mean()}

        left, right, best_val = split_data(X, y, best_attr)
        # print('left X:',left[0],'\nleft Y:',left[1],'\nright X:',right[0],'\nright y:',right[1])

        if left[0].empty : # type: ignore
            # print("left is empty")
            l_st = {"leaf": True, "split_value": y.mean() if(len(left[0])) else 0}
        else: 
            l_st = self.build_tree_diro(left[0], left[1], depth+1) # type: ignore
        if right[0].empty: # type: ignore
            r_st =  {"leaf": True, "split_value":  y.mean() if(len(right[0])) else 0}
        else:
            r_st = self.build_tree_diro(right[0], right[1], depth+1) # type: ignore

        # print({"leaf": False, "split_attribute": best_attr, "split_value": best_val,
        #         "left_subtree": l_st, "right_subtree": r_st})

        return {"leaf": False, "split_attribute": best_attr, "split_value": best_val,
                "left_subtree": l_st, "right_subtree": r_st}
    
    def build_tree_rido(self, X: pd.DataFrame, y: pd.Series, depth ):
        if depth == self.max_depth or len(y.unique()) == 1 :
            # print('Y behaviour:',y,'\n')
            # print({"leaf": True, "split_value": np.bincount(y).argmax()})
            return {"leaf": True, "split_value": np.bincount(y).argmax()}
        
        best_attr = opt_split_attribute(X, y, criterion='', features= X.columns) # type: ignore

        if best_attr is None:
            # print( {"leaf": True, "split_value": y.value_counts().argmax()})
            return {"leaf": True, "split_value": y.value_counts().argmax()}

        left, right, best_val = split_data(X, y, best_attr)
        # print('left X:',left[0],'\nleft Y:',left[1],'\nright X:',right[0],'\nright y:',right[1])

        l_st = self.build_tree_rido(left[0], left[1], depth+1) # type: ignore
        r_st = self.build_tree_rido(right[0], right[1], depth+1) #type: ignore

        # print({"leaf": False, "split_attribute": best_attr, "split_value": best_val,
        #         "left_subtree": l_st, "right_subtree": r_st})

        return {"leaf": False, "split_attribute": best_attr, "split_value": best_val,
                "left_subtree": l_st, "right_subtree": r_st}

    def build_tree_riro(self, X: pd.DataFrame, y: pd.Series, depth ):

        if depth == self.max_depth or ((y - y.mean())**2).mean() == 0 :
            # print('Y behaviour:',y,'\n')
            # print({"leaf": True, "split_value": y.mean()})
            return {"leaf": True, "split_value": y.mean()}
        
        best_attr = opt_split_attribute(X, y, criterion='', features= X.columns) # type: ignore

        if best_attr is None:
            # print( {"leaf": True, "split_value": y.mean()})
            return {"leaf": True, "split_value": y.mean()}

        left, right, best_val = split_data(X, y, best_attr)
        # print('left X:',left[0],'\nleft Y:',left[1],'\nright X:',right[0],'\nright y:',right[1])

        if left[0].empty : # type: ignore
            # print("left length is 0")
            l_st = {"leaf": True, "split_value": y.mean() if(len(left[0])) else 0}
        else: 
            l_st = self.build_tree_riro(left[0], left[1], depth+1) # type: ignore
        if right[0].empty: # type: ignore
            r_st =  {"leaf": True, "split_value":  y.mean() if(len(right[0])) else 0}
        else:
            r_st = self.build_tree_riro(right[0], right[1], depth+1) # type: ignore

        # print({"leaf": False, "split_attribute": best_attr, "split_value": best_val,
        #         "left_subtree": l_st, "right_subtree": r_st})

        return {"leaf": False, "split_attribute": best_attr, "split_value": best_val,
                "left_subtree": l_st, "right_subtree": r_st}



    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 
        xtype = (X.dtypes == "category").all()
        ytype = check_ifreal(y)

        self.xtype = xtype
        self.ytype = ytype

        if(self.xtype and self.ytype):
            print('Discrete input discrete output------------')
            self.tree = self.build_tree_dido(X,y,0) #done

        if(self.xtype and not(self.ytype)):
            print('Discrete input real output----------------')
            self.tree = self.build_tree_diro(X,y,0) 
            

        if(not(self.xtype) and self.ytype):
            print('Real input discrete output----------------')
            self.tree = self.build_tree_rido(X,y,0)

        if(not(self.xtype) and not(self.ytype)):
            print('Real input real output--------------------')
            self.tree = self.build_tree_riro(X,y,0) #done
    

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        # xtype = check_ifreal(X[0])

        y_hat = []

        # print('self tree:',self.tree)
        # print('length of X:',len(X))
        # if tree_copy is not None:
        #     print('Condition', X[tree_copy['split_attribute']][0] == tree_copy['split_value'])

        if((self.xtype and self.ytype) or (self.xtype and not(self.ytype))):
            'Discrete input discrete output & Discrete input real output'
            for i in range(len(X)):
                # print('I value:',i)
                tree_copy = self.tree
                leaf = False
                while not(leaf) and tree_copy is not None:
                    # print('tree_copy[split_attribute]:',tree_copy['split_attribute'])
                    attribute = tree_copy['split_attribute']
                    value = tree_copy['split_value']
                    # print('Tree copy inside while:',tree_copy['split_attribute'],'X[attribute][i]:',X[attribute][i])
                    if (X[attribute][i] == value):
                        tree_copy = tree_copy['left_subtree']
                        # print('tree_copy:',tree_copy)
                        leaf = tree_copy['leaf']
                        # print('leaf:',leaf)
                    else:
                        tree_copy = tree_copy['right_subtree']
                        # print('tree_copy:',tree_copy)
                        leaf = tree_copy['leaf']
                        # print('leaf:',leaf)
                
                if leaf == True and tree_copy is not None:
                        y_hat.append(tree_copy['split_value'])

        if((not(self.xtype) and not(self.ytype)) or (not(self.xtype) and self.ytype)):
            'Real input real output & Real input discrete output'
            for i in range(len(X)):
                # print('I value:',i)
                tree_copy = self.tree
                leaf = False
                while not(leaf) and tree_copy is not None:
                    # print('tree_copy[split_attribute]:',tree_copy['split_attribute'])
                    attribute = tree_copy['split_attribute']
                    value = tree_copy['split_value']
                    # print('Tree copy inside while:',tree_copy['split_attribute'],'X[attribute][i]:',X[attribute][i])
                    if (X[attribute][i] <= value):
                        tree_copy = tree_copy['left_subtree']
                        # print('tree_copy:',tree_copy)
                        leaf = tree_copy['leaf']
                        # print('leaf:',leaf)
                    else:
                        tree_copy = tree_copy['right_subtree']
                        # print('tree_copy:',tree_copy)
                        leaf = tree_copy['leaf']
                        # print('leaf:',leaf)
                
                if leaf == True and tree_copy is not None:
                        y_hat.append(tree_copy['split_value'])

        y_hat = pd.Series(y_hat)
        return y_hat

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
        # print(self.tree)
        dot = self.plot_tree(self.tree)
        if(self.xtype and self.ytype):
            dot.render('DecisionTree_dido', format='png', cleanup=True)

        if(self.xtype and not(self.ytype)):
            dot.render('DecisionTree_diro', format='png', cleanup=True)

        if(not(self.xtype) and self.ytype):
            dot.render('DecisionTree_rido', format='png', cleanup=True)
        
        if(not(self.xtype) and not(self.ytype)):
            dot.render('DecisionTree_riro', format='png', cleanup=True)
        
        

        

    def plot_tree(self, tree, dot=None, parent_name=None, edge_label=None):
        node_label = ''
        if dot is None:
            dot = graphviz.Digraph(comment='Decision Tree')

        if 'leaf' in tree and tree['leaf']:
            leaf_label = f"Value: {tree['split_value']:.4f}"
            dot.node(str(id(tree)), leaf_label, shape='box')
            if parent_name is not None:
                dot.edge(parent_name, str(id(tree)), label=edge_label)
        else:
            if(not(self.xtype) and not(self.ytype)):
                node_label = f"Feature {tree['split_attribute']} <= {tree['split_value']:.4f}"
            if(self.xtype and self.ytype):
                node_label = f"Feature {tree['split_attribute']} = {tree['split_value']:.4f}"
            if(self.xtype and not(self.ytype)):
                node_label = f"Feature {tree['split_attribute']} = {tree['split_value']:.4f}"
            if(not(self.xtype) and self.ytype):
                node_label = f"Feature {tree['split_attribute']} <= {tree['split_value']:.4f}"

            dot.node(str(id(tree)), node_label)

            if parent_name is not None:
                dot.edge(parent_name, str(id(tree)), label=edge_label)

            left_edge_label = 'Yes'
            right_edge_label = 'No'

            self.plot_tree(tree['left_subtree'], dot, str(id(tree)), left_edge_label)
            self.plot_tree(tree['right_subtree'], dot, str(id(tree)), right_edge_label)

        return dot
