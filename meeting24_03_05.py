#Diffusion Theory

#Regression - Linear - Polynomial - Lasso - Ridge - Poisson - Quantile - Logistic

#Validation and Metrics

#Coding
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_tweedie_deviance, PredictionErrorDisplay
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

#DATA EXTRACTION AND FORMATTING

data = pd.read_csv('AqueousDiffusion.csv')
x = data.drop('Λ_eq', axis = 1)
x = data.drop('Anion/Cation', axis = 1)
y = data['Λ_eq']

pca = decomposition.PCA(n_components=3)
pca.fit(x)
x = pca.transform(x)

#x['z'] = pd.Series(map(lambda n: n**(2), x[1]))
#nx = x[0]*x[1]

print(x)
#print(y)

Xtr, Xte, ytr, yte = train_test_split(x, y, test_size = 0.8, random_state=104)
#scaler = StandardScaler()
#Xtr =scaler.fit_transform(Xtr)
#ytr =scaler.fit_transform(ytr)

#Xtr = np.array(Xtr)
#Xte = np.array(Xte)
#Xtr = Xtr.reshape(-1, 1)
#Xte = Xte.reshape(-1, 1)
#print(yte)
yTe = np.array(yte)
#print(yTe)
#print(Xtr)

#print(Xtr)
#print(ytr)
#print(Xte)
#print(yte)



#MODEL

regRidge = linear_model.Lasso(alpha = 0.05) 
regRidge.fit(Xtr, ytr)
pred = regRidge.predict(Xte)
print(pred)

#VALIDATION



#METRICS

#r2 = r2_score(yte, pred)
#print(r2)
#mse = mean_squared_error(yte, pred)
##print(mse)
#mae= mean_absolute_error(yte, pred)
#print(mae)
rms = mean_squared_error(yte, pred, squared = False)
print(rms)
r2 = r2_score(yte, pred)
print(r2)
#mtd = mean_tweedie_deviance(yte, pred, power = 2)
#print(mtd)

#PLOTTING

plt.cla()
plt.clf()

#fig = plt.figure(figsize =(5, 5))

#disp = PredictionErrorDisplay(y_true = yTe,y_pred = pred)
disp = PredictionErrorDisplay.from_predictions(y_true = yTe,y_pred = pred, kind="actual_vs_predicted")
disp.plot()

#ax = fig.add_subplot(1, 1, 1)
#ax.set_xscale('linear')
#ax.set_xscale('linear')
# plt.title('Diffusion Coeff. vs MM*z')
# plt.legend('Blue Dots')
# plt.xticks()
# plt.yticks()
# plt.xlabel('z^2D')
# plt.ylabel('Molar Conduct.')
# plt.grid(True)
# plt.scatter(nx, y, color = 'blue', marker = '.', s = 2)

plt.show()
