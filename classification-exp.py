import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

'Real input discrete output'
# print(type(X))
# print(X)
# print(len(X))
# print(type(y))
# print(y)
# print(len(y))
# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

X = pd.DataFrame(X)
y = pd.Series(y, dtype = 'category')

end = int(0.7*len(X))
# 70% training data
X_train = X[ : end]
y_train = y[ : end]

# 30% test data 
X_test = X[end : ]
X_test = X_test.reset_index(drop = True)
y_test = y[end : ]
y_test = y_test.reset_index(drop = True)

tree = DecisionTree(criterion='information_gain')  # Split based on Inf. Gain
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)
print("Accuracy: ", accuracy(y_hat, y_test))
for cls in y_test.unique():
        print("Precision: ", precision(y_hat, y_test, cls))
        print("Recall: ", recall(y_hat, y_test, cls))


#------------------------------5 fold cross validation-----------------------------------------

k = 5

# Initialize lists to store predictions and accuracies
predictions = {}
accuracies = []

# Calculate the size of each fold
fold_size = len(X) // k

# Perform k-fold cross-validation
for i  in range(k):
    # Split the data into training and test sets
    test_start = i * fold_size
    test_end = (i + 1) * fold_size
    test_set = X[test_start:test_end]
    test_set = test_set.reset_index(drop = True)
    test_labels = y[test_start:test_end]
    test_labels = test_labels.reset_index(drop = True)
    
    # Train the model
    dt_classifier = DecisionTree(criterion='information_gain',max_depth=4)
    # dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(test_set, test_labels)
    
    # Make predictions on the validation set
    fold_predictions = dt_classifier.predict(test_set)
    
#     # Calculate the accuracy of the fold
    fold_accuracy = np.mean(fold_predictions == test_labels)
    
#     # Store the predictions and accuracy of the fold
    predictions[i] = fold_predictions
    accuracies.append(fold_accuracy)

# # Print the predictions and accuracies of each fold
print('Five fold Cross Validation output')
for i in range(k):
     print(" the fold {}: accuracy: {:.4f}".format(i+1, accuracies[i]))

#----------------5 fold Nested Cross Validation-------------------------------------------------
# Define the number of folds (k)
k = 5

# Initialize lists to store predictions and accuracies
predictions = {}
accuracies = []
results = {}
overall_count = 0
# Calculate the size of each fold
fold_size = len(X) // k

# Perform k-fold cross-validation
for i in range(k): # outer loop for train test split
    # Split the data into training and test sets
    test_start = i * fold_size
    test_end = (i + 1) * fold_size

    test_set = X[test_start:test_end]
    test_labels = y[test_start:test_end]
    test_set = test_set.reset_index(drop = True)
    test_labels = test_labels.reset_index(drop = True)

    
    training_val_set = pd.concat((X[:test_start], X[test_end:]), axis=0)
    # print(training_set)
    training_val_labels = pd.concat((y[:test_start], y[test_end:]), axis=0)
    training_val_set = training_val_set.reset_index(drop = True)
    training_val_labels = training_val_labels.reset_index(drop = True)


    # Calculate the size of each fold
    fold_size = len(training_val_set) // k

    for j in range(k):
        # split the data into training and validation set
        val_start = j * fold_size
        val_end = (i + 1) * fold_size

        val_set = training_val_set[val_start:val_end]
        val_labels = training_val_labels[val_start:val_end]
        val_set = val_set.reset_index(drop = True)
        val_labels = val_labels.reset_index(drop = True)

    
        training_set = pd.concat((training_val_set[:val_start], training_val_set[val_end:]), axis=0)
        # print(training_set)
        training_labels = pd.concat((training_val_labels[:val_start], training_val_labels[val_end:]), axis=0)
        training_set = training_set.reset_index(drop = True)
        training_labels = training_labels.reset_index(drop = True)

        depth = range(1,7)
        for d in depth:
            # Train the model
            dt_classifier = DecisionTree(criterion = 'information_gain',max_depth= d)
            dt_classifier.fit(training_set, training_labels)
        
            # Make predictions on the validation set
            fold_predictions = dt_classifier.predict(val_set)
        
            # Calculate the accuracy of the fold
            val_accuracy = np.mean(fold_predictions == val_labels)

            results[overall_count] = {'outer_fold': i, 
                                      'inner_fold': j, 
                                      'max_depth': d, 
                                      'val_accuracy': val_accuracy}
            overall_count += 1

# Print the accuracies 
overall_results = pd.DataFrame(results).T
print(overall_results)

# Optimal depth of tree is 2
print(overall_results.groupby(['max_depth']).mean()['val_accuracy'].sort_values(ascending=False).head(10))