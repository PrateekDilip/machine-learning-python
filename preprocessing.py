# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

#Indep/dep variables
#categorical variables
#Training set
#test set 
#Missing data
#Feature Scaling
"""
Spyder Editor

This is a temporary script file.
"""
#import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as scy
from pathlib import Path

print("Hello World!")


#import the dataset
dataset = pd.read_csv("/Users/prateekd/Downloads/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv")

#create matrix of independent variables
X = dataset.iloc[:, :-1].values

#create dependent variable vecrtors
Y = dataset.iloc[:,3].values

X
Y

#Take care of missing data in dataset
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])



#Take care of categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()


labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Split dataset in Trainingset and Testset
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

