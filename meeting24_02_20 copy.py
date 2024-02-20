#Diffusion Theory

#Regression - Linear - Polynomial - Lasso - Ridge - Poisson - Quantile - Logistic

#Validation and Metrics

#Coding
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import pandas as pd

data = pd.read_csv('AqueousDiffusion.csv')
x = [data['Molar Mass'], data['z']]
y = data['D_i/10^-9 [m^2/s]']

plt.scatter(x[0], y)
plt.plot()
