"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np
from metrics import *

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if(y.dtype == 'category' ):
        return True
    else:
        return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy -p(log p)
    """
    prob = Y.value_counts(normalize=True)
    entropy = np.sum(-prob*np.log2(prob+1e-9))
    return entropy

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    y = Y.value_counts(normalize=True)
    return (y*(1-y)).sum()


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain (information gain from each attribute)
    """
    ytype = check_ifreal(Y)
    xtype = check_ifreal(attr)
    # print('xtype ',xtype)

    info_gain = 0

    if(xtype and not(ytype)):
        'discrete input real output'
        prob = attr.value_counts(normalize=True)
        attr_mse = {}
        for category in attr.unique():
            index = attr == category
            attr_mse[category] = ((Y[index] - Y[index].mean())**2).mean()
        attr_mse = pd.Series(attr_mse)   
        info_gain =((Y - Y.mean())**2).mean()  - np.sum(prob*attr_mse)

    elif(ytype):
        'discrete output'
        prob = attr.value_counts(normalize=True)
        attr_entropy = {}
        for category in attr.unique():
            index = attr == category
            attr_entropy[category] = entropy(Y[index])

        attr_entropy = pd.Series(attr_entropy)   
        info_gain = entropy(Y) - np.sum(prob*attr_entropy)

    elif(not(ytype)):
        'Real output'
        min_mse = 999999
        # print('Attribute entering information gain: ',attr)
        for value in attr.unique():
            # print('Value:',value)

            left = Y[attr <= value]
            # print('left',left)
            right = Y[attr > value]
            # print('right:',right)

            left_mean = left.mean() if len(left) > 0 else 0
            # print('left_mean:',left_mean)
            right_mean = right.mean() if len(right) > 0 else 0
            # print('rigth_mean:',right_mean)
            
            left_mse = ((left - left_mean)**2).mean() if len(left) > 0 else 0
            right_mse = ((right - right_mean)**2).mean() if len(right) > 0 else 0
            combined_mse = left_mse + right_mse
            # print('combined_mse ',combined_mse)

            if combined_mse <=min_mse:
                min_mse = combined_mse

        info_gain = min_mse
        # print('Minimum RMSE', info_gain)
        
    return info_gain


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    
    ytype = check_ifreal(y)

    best_attr = None

    if features.empty:
        return best_attr
    
    xtype = check_ifreal(X[features[0]])

    if(xtype and not(ytype)):
        'Discrete input real output'
        info_gain = {}
        for attribute in features:
            print('attribute number',attribute)
            info_gain[attribute] = information_gain(y, X[attribute])
            print('RMSE :',info_gain[attribute])
        info_gain = pd.Series(info_gain)
        print('Info gain in series:',info_gain.idxmin())
        best_attr = info_gain.idxmax()

    
    elif(ytype):
        'Discrete input discrete output'
        info_gain = {}
        for attribute in features:
            # print('attribute number',attribute)
            info_gain[attribute] = information_gain(y, X[attribute])
            # print(info_gain[attribute])
        info_gain = pd.Series(info_gain)
        # print('Info gain in series:',info_gain.idxmax())
        best_attr = info_gain.idxmax()

    elif(not(ytype)):
        'Real input real output'
        info_gain = {}
        for attribute in features:
            # print('attribute number',attribute)
            info_gain[attribute] = information_gain(y, X[attribute])
            # print('RMSE :',info_gain[attribute])
        info_gain = pd.Series(info_gain)
        # print('Info gain in series:',info_gain.idxmin())
        best_attr = info_gain.idxmin()


    print("Best attribute:", best_attr)
        
    return best_attr


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value = 0):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    xtype = check_ifreal(X[attribute])
    ytype = check_ifreal(y)

    best_val = -1
    X_left,y_left,X_right, y_right = [],[],[],[]

    if(xtype and not(ytype)):
        'Discrete input real output'
        min_mse = 99999
        for value in X[attribute].unique():
            index = X[attribute] == value
            target_var = y[index]
            attr_mse = ((target_var - target_var.mean())**2).mean()
            if attr_mse < min_mse:
                min_mse = attr_mse
                best_val = value
        left_mask = X[attribute] == best_val
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        print('best value in split data:',best_val)

    elif(ytype):
        'Discrete input discrete output'
        min_entropy = 9999
        for value in X[attribute].unique():
            index = X[attribute] == value
            target_var = y[index]
            val_entropy = entropy(target_var)
            if val_entropy < min_entropy:
                min_entropy = val_entropy
                best_val = value

        left_mask = X[attribute] == best_val
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        # print('best value:',best_val)

    elif(not(ytype)):
        'Real input real output'
        min_mse = 999999
        # print('Attribute in split data:',attribute)
        for value in X[attribute].unique():
            left = y[X[attribute] <= value]
            right = y[X[attribute] > value]
            
            left_mean = left.mean() if len(left) > 0 else 0
            right_mean = right.mean() if len(right) > 0 else 0

            left_mse = ((left - left_mean)**2).mean() if len(left) > 0 else 0
            right_mse = ((right - right_mean)**2).mean() if len(right) > 0 else 0
            combined_mse = left_mse + right_mse

            # print('combined_mse ',combined_mse)

            if combined_mse <= min_mse:
                min_mse = combined_mse
                best_val = value   
        
        left_mask = X[attribute] <= best_val
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        # print('best value in split data:',best_val)

    return (X_left, y_left), (X_right, y_right), best_val


