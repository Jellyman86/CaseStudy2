# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2#importing libraries under standard aliases
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#inline visualizations


#importing our models and scoring libraries
#we'll be using mostly Scikit.Learn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

#visualtion methods and methods for model analysis
from sklearn.metrics import roc_curve, auc, confusion_matrix

#setting a random seed for reproduceability
import random
random.seed(42)

#setting filter warnings to ignore to keep our notebook clean
import warnings
warnings.filterwarnings("ignore")

#loading the whole data set from the csv file and into pandas
#note that this file is ';' separated, not ',' separated

data = pd.read_csv("bank-additional-full.csv", sep = ';')

#reviwing the dataframe to ensure everything loaded correctly
data.head()


print(data.shape)

print(data.describe())


#checking for Null values 
print(data.isna().any())


#looking at our target column 
print(data.y.unique())


for i in range(len(data.y)):
    if data.y[i] == 'no':
        data.y[i] = 0
    else:
        data.y[i]=1
        
print(data.y.unique())


print(data.y.sum())
print(np.__version__)
print(data.info())