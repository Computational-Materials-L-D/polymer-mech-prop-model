from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


###################################
#Data Preparation

df = pd.read_csv('Data.csv')
df['Material'] = df[['Std', 'Material', 'Heat treatment']].fillna('').agg(' '.join, axis = 1)
#print(df['Material'])
df.drop(['Heat treatment','G','ID','Material','Std','Sy','A5','Bhn','mu','pH','Desc','HV'], axis = 1, inplace = True)

df.to_csv('regRo_Su', index = False)

################################
#Data Training and Testing Set / features and labels

df0 = pd.read_csv('regRo_Su')

# sklearn -> train_test_split()
df0 = df0.sample(frac = 1, shuffle = True)
train = df0.sample(frac = 0.8)
test = df0.drop(train.index)

X_tr = train.loc[:, train.columns !='Su']
Y_tr = train.loc[:, train.columns =='Su']

X_te = test.loc[:, test.columns !='Su']
Y_te = test.loc[:, test.columns =='Su']

Xtr, Xte, ytr, yte = train_test_split(df0, test_size = 0.25)

###############################
#Model Parameters

reg = linear_model.Ridge(alpha = 1)
reg.fit(X_tr, Y_tr)

reg1 = linear_model.Ridge(alpha = 1)
reg1.fit(Xtr, ytr)

################################
#Model Testing

Y_p = reg.predict(X_te)
yp= reg.predict(Xte)
#print(Y_te)
#print(X_te)

print(mean_squared_error(Y_te, Y_p))
print(np.sqrt(mean_squared_error(Y_te, Y_p)))
print(r2_score(Y_te, Y_p))

print(mean_squared_error(Y_te, yp))
print(np.sqrt(mean_squared_error(Y_te, Y_p)))
print(r2_score(Y_te, Y_p))

cv = cross_validate(reg1, Xte, yte, cv = 4, return_estimator = True)
print(cv)
print(cv['test_score'])
print(np.mean(cv['test_score']))

gtest_score = []
#for i in range(len(cv_results['estimator'])):
#  val_score.append(cv_results['estimator'][i].score(X_gtest, y_gtest))
#sum(gtest_score) / len(gtest_score)
###############################
#Model Visualization

plt.scatter(Y_te, Y_p, color = 'g')
plt.plot([1,2000], [1,2000], '--')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.xticks()
plt.yticks()
# Line of Best Fit
m, b = np.polyfit(Y_te, Y_p, 1)
plt.plot(Y_te, m * Y_te + b, color='r')


plt.show()