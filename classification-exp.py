# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from tree.base import DecisionTree
# from metrics import *
# from sklearn.datasets import make_classification

# np.random.seed(42)

# # Code given in the question
# X, y = make_classification(
#     n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# # For plotting
# plt.scatter(X[:, 0], X[:, 1], c=y)

# # Write the code for Q2 a) and b) below. Show your results.


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

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import  KFold
from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score

# Generate dataset
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5
)

# Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X)
print(y)
print(len(X))
print(int(0.7 * len(X)))
index = int(0.7 * len(X))

X = pd.DataFrame(X)
y = pd.Series(y, dtype = 'category')

#training data
X_train = X[ : index]
print(X_train)
y_train = y[ : index]
print(y_train)

X_test = X[index : ]
X_test = X_test.reset_index(drop = True)
print(X_test)

y_test = y[ index : ]
y_test = y_test.reset_index(drop = True)
print(y_test)


# Create a decision tree classifier
clf = DecisionTree(criterion='information_gain')

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate performance
accuracy = accuracy(y_test, y_pred)

for y_class in y_test.unique():
    print('Precision of ', y_class,'is',precision(y_pred, y_test, y_class))  # Per-class precision
    print('Recall of ',y_class,'is', recall(y_pred, y_test, y_class))  # Per-class recall

print("Accuracy:", accuracy)
print("===========================================")


# Nested cross-validation to find optimal depth
# param_grid = {'max_depth': range(1, 11)}
# grid_search = GridSearchCV(clf, param_grid, cv=5)
# grid_search.fit(X, y)

# best_depth = grid_search.best_params_['max_depth']
# print("Optimal depth:", best_depth)

#---------------
import numpy as np
# Define the number of folds (k)
k = 5

# Initialize lists to store predictions and accuracies
predictions = {}
accuracies = []

# Calculate the size of each fold
fold_size = len(X) // k

# Perform k-fold cross-validation
for i  in range(k):
    # Split the data into training and test sets

    # y = pd.Series(y, dtype = 'category')
    test_start = i * fold_size
    test_end = (i + 1) * fold_size
    test_set = X[test_start:test_end]
    test_set = test_set.reset_index(drop = True)
    print("%%%%%%%%%%%%%%%test set",test_set)
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
for i in range(k):
     print(" the fold {}: accuracy: {:.4f}".format(i+1, accuracies[i]))
#---








