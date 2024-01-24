import os
import tsfel
import IPython
import warnings
import numpy as np
import pandas as pd
cfg = tsfel.get_features_by_domain()
from sklearn.decomposition import PCA
pca_acceleration = PCA(n_components=2)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore") 


# Training Data

time = 10
offset = 100
folders = ["LAYING","SITTING","STANDING","WALKING","WALKING_DOWNSTAIRS","WALKING_UPSTAIRS"]
classes = {"WALKING":1,"WALKING_UPSTAIRS":2,"WALKING_DOWNSTAIRS":3,"SITTING":4,"STANDING":5,"LAYING":6}

combined_dir = os.path.join("Deployment")

count_files = 0
for folder in folders:
    files = os.listdir(os.path.join(combined_dir,"Train",folder))
    for file in files:
        count_files += 1

X_train_tsfel = []
y_train_tsfel = []

dataset_dir = os.path.join(combined_dir,"Train")


for folder in folders:
    files = os.listdir(os.path.join(dataset_dir,folder))

    for file in files:
        acc_data_tsfel = [] 
        print("Remaining Files: ", count_files)
        count_files -= 1    
        df = pd.read_csv(os.path.join(dataset_dir,folder,file),sep=",",header=0)
        df = df[offset:offset+time*50]
        df["linacc"] = df["accx"]**2 + df["accy"]**2 + df["accz"]**2

        acc_data_tsfel.append(df["linacc"].values)

        tsfel_features = tsfel.time_series_features_extractor(cfg, acc_data_tsfel, verbose=1)
        IPython.display.clear_output()
        
        X_train_tsfel.append(tsfel_features)
        y_train_tsfel.append(classes[folder])

        
print(tsfel_features.shape)
X_train_tsfel = np.array(X_train_tsfel).reshape(180, 384)
print(X_train_tsfel.shape)
y_train_tsfel = np.array(y_train_tsfel)
print(y_train_tsfel.shape)

# Testing Data

count_files = 0
for folder in folders:
    files = os.listdir(os.path.join(combined_dir,"Test",folder))
    for file in files:
        count_files += 1

X_test_tsfel = []
y_test_tsfel = []

dataset_dir = os.path.join(combined_dir,"Test")


for folder in folders:
    files = os.listdir(os.path.join(dataset_dir,folder))

    for file in files:
        acc_data_tsfel = [] 
        print("Remaining Files: ", count_files)
        count_files -= 1    
        df = pd.read_csv(os.path.join(dataset_dir,folder,file),sep=",",header=0)
        # df = df[offset:offset+time*200:4]
        df = df[offset:offset+time*50]
        # df["linacc"] = df.iloc[:, 1]**2 + df.iloc[:,2]**2 + df.iloc[:,3]**2
        df["linacc"] = df.iloc[:, 0]**2 + df.iloc[:,1]**2 + df.iloc[:,2]**2

        acc_data_tsfel.append(df["linacc"].values)

        tsfel_features = tsfel.time_series_features_extractor(cfg, acc_data_tsfel, verbose=1)
        IPython.display.clear_output()
        
        X_test_tsfel.append(tsfel_features)
        y_test_tsfel.append(classes[folder])

print(tsfel_features.shape)
X_test_tsfel = np.array(X_test_tsfel).reshape(30, 384)
print(X_test_tsfel.shape)
y_test_tsfel = np.array(y_test_tsfel)
print(y_test_tsfel.shape)
print(y_test_tsfel)

X = np.concatenate((X_train_tsfel,X_test_tsfel))
y = np.concatenate((y_train_tsfel,y_test_tsfel))

# Splitting the data into training and testing sets
X_train_tsfel,X_test_tsfel,y_train_tsfel,y_test_tsfel = train_test_split(X,y,test_size=0.4,random_state=4,stratify=y)
print("Training data shape: ",X_train_tsfel.shape)
print("Testing data shape: ",X_test_tsfel.shape)
print("Training data shape: ",y_train_tsfel.shape)
print("Testing data shape: ",y_test_tsfel.shape)