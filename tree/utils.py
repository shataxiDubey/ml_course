"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import numpy as np
import pandas as pd


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if pd == "category" or pd == "object":
        return False
    return True


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    value_counts = Y.value_counts() # Get unique values and their count
    probabilities = value_counts / len(Y) # For all unique values' count calc. prob.
    entropy_value = -np.sum(probabilities * np.log2(probabilities)) # Shorthand for entropy calc.

    return entropy_value


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    value_counts = Y.value_counts()
    probabilities = value_counts / len(Y)
    gini_value = 1 - np.sum(probabilities * probabilities)

    return gini_value


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    if len(Y) == 0:
        return 0
    
    if attr == "category" or attr == "object":
        attr = attr.astype("category")
        inner_class_counts = attr.value_counts()
        inner_class_proportions = inner_class_counts/len(Y)
        entropy_classes = []
        for i in inner_class_counts.index:
            entropy_classes.append(entropy(attr==i))

        information_gain_val = entropy(Y) - np.sum(inner_class_proportions * entropy_classes)

    return information_gain_val


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
