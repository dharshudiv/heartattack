import numpy as np
from urllib.request import urlopen
import urllib
import matplotlib.pyplot as plt # Visuals
import seaborn as sns 
import sklearn as skl
import pandas as pd

from sklearn.cross_validation import train_test_split # Create training and test sets
from sklearn.neighbors import KNeighborsClassifier # Kth Nearest Neighbor
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.tree import export_graphviz # Extract Decision Tree visual
from sklearn.tree import tree 
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn import svm #SVM
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # AUC 
from sklearn.model_selection import KFold, cross_val_score #cross validation 
from sklearn import cross_validation  #cross validation 
from urllib.request import urlopen # Get data from UCI Machine Learning Repository

import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as pt
plt.style.use('ggplot')
pt.set_credentials_file(username='rboone125', api_key='')
Cleveland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
Hungarian_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
Switzerland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data'
np.set_printoptions(threshold=np.nan) #see a whole array when we output it

names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
ClevelandHeartDisease = pd.read_csv(urlopen(Cleveland_data_URL), names = names) #gets Cleveland data
HungarianHeartDisease = pd.read_csv(urlopen(Hungarian_data_URL), names = names) #gets Hungary data
SwitzerlandHeartDisease = pd.read_csv(urlopen(Switzerland_data_URL), names = names) #gets Switzerland data
datatemp = [ClevelandHeartDisease, HungarianHeartDisease, SwitzerlandHeartDisease] #combines all arrays into a list

heartDisease = pd.concat(datatemp)#combines list into one array
heartDisease.head()
del heartDisease['ca']
del heartDisease['slope']
del heartDisease['thal']

heartDisease = heartDisease.replace('?', np.nan)
heartDisease.dtypes
#loop to count the "?" per variable. More for interpretation. 

count = 0
for item in heartDisease:
    for i in heartDisease[item]:
        count += (i == '?')
        def classImbalance(item):
    item_0 = 0
    item_1 = 0
    item_2 = 0
    item_3 = 0
    item_4 = 0
    item_5 = 0
    for item_i in heartDisease[item]:
        for i in range(0,6):
            if (item == i):
                item_i +=1
    heartDisease_i = 0
    for i in  range (0,6):
        heartDisease_i = (item_i/len(heartDisease)) * 100
        print("The percentage of level", i, "in the response variable is: {0:.2f}".format(heartDisease_i)) 

classImbalance('heartdisease')
trace0 = go.Box(
    y=heartDisease['age'],
    name='age'
)
trace1 = go.Box(
    y=heartDisease['sex'],
    name='sex'
)
trace2 = go.Box(
    y=heartDisease['cp'],
    name='cp'
)
trace3 = go.Box(
    y=heartDisease['trestbps'],
    name='trestbps'
)
trace4 = go.Box(
    y=heartDisease['chol'],
    name='chol'
)
trace5 = go.Box(
    y=heartDisease['fbs'],
    name='fbs'
)
trace6 = go.Box(
    y=heartDisease['restecg'],
    name='restecg'
)
trace7 = go.Box(
    y=heartDisease['thalach'],
    name='thalach'
)
trace8 = go.Box(
    y=heartDisease['exang'],
    name='exang'
)
trace9 = go.Box(
    y=heartDisease['oldpeak'],
    name='oldpeak'
)
trace10 = go.Box(
    y=heartDisease['heartdisease'],
    name='heart disease status'
)
plotdata = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10]
py.iplot(plotdata)
