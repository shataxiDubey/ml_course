# from os import P_ALL
from itertools import count
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)

# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs

def run_time_complexity(case, maxdepth):

    count = 100
    n_array=[]
    m_array=[]
    fit_time=[]
    predict_time=[]

    for n in range(30,40):
        for m in range(5,15):
            n_array.append(n)
            m_array.append(m)

            print(count," Iteration Remaining")
            

            if(case=="DD"):
                #Discrete input discrete output
                X = pd.DataFrame(np.random.randn(n, m))
                y = pd.Series(np.random.randn(n))
                label="DISCRETE INPUT DISCRETE OUTPUT"

            if(case=="DR"):
                #discrete input real output
                X = pd.DataFrame({i: pd.Series(np.random.randint(m, size=n), dtype="category") for i in range(2)})
                y = pd.Series(np.random.randn(n))
                label="DISCRETE INPUT REAL OUTPUT"

            if(case=="RD"):
                #real input discrete output
                X = pd.DataFrame(np.random.randn(n, m))
                y = pd.Series(np.random.randint(m, size=n), dtype="category")
                label="REAL INPUT DISCRETE OUTPUT"

            if(case=="RR"):
                #real input real output
                X = pd.DataFrame(np.random.randn(n, m))
                y = pd.Series(np.random.randn(n))
                label="REAL INPUT REAL OUTPUT"

            tree=DecisionTree(criterion='information_gain',max_depth=maxdepth)
            
            temp = num_average_time
            starttime=time.time()
            for _ in range(num_average_time):
                #time for fitting
                print(count,"Iteration and", temp,"internal Fit Iteration Remaining")
                temp -= 1
                tree.fit(X,y)   
            
            endtime=time.time()

            fit_time.append((endtime - starttime)/num_average_time)

            temp = num_average_time
            starttime=time.time()
            for _ in range(num_average_time):
                #time for prediction
                print(count,"Iteration and", temp,"internal Predition Iteration Remaining")
                y_hat=tree.predict(X)
            
            endtime=time.time()
            
            predict_time.append((endtime-starttime)/num_average_time)
            
            count -= 1

    scatter_plot(n_array, m_array, fit_time, label+" Depth "+str(maxdepth)+' Fit')
    scatter_plot(n_array, m_array, fit_time, label+" Depth "+str(maxdepth)+' Predict')

# Function to plot the results

def scatter_plot(n_array, m_array, time, label):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(n_array, m_array, time)
    ax.set_xlabel("N")
    ax.set_ylabel("M")
    ax.set_zlabel("Time")
    plt.title(label)   
    plt.savefig("./Experiments/"+label+".png")

# Other functions
# ...
    
# Run the functions, Learn the DTs and Show the results/plots
    
run_time_complexity("DD", 5)
run_time_complexity("DR", 5)
run_time_complexity("RR", 5)
run_time_complexity("RD", 5)
run_time_complexity("DD", 9)
run_time_complexity("DR", 9)
run_time_complexity("RR", 9)
run_time_complexity("RD", 9)


