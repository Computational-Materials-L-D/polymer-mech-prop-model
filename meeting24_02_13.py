#Regression - Linear - Polynomial - Lasso - Ridge - Poisson - Quantile - Logistic
#Coding
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import pandas as pd

data = pd.read_csv('material.csv')
#print(data)

ro = data['Ro']
e = data['E']
data = [ro, e]

model = linear_model.Ridge()
model.fit(ro, e)

Xtr, Xte, ytr, yte = train_test_split(data, test_size = 0.25)

