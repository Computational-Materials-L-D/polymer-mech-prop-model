#Validation (cross) and dimensionality reduction 

#Coding
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import linear_model, decomposition, datasets
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, mean_absolute_error, mean_tweedie_deviance, PredictionErrorDisplay, make_scorer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler


#DATA EXTRACTION AND FORMATTING

data = pd.read_csv('AqueousDiffusion.csv')
y = data['Λ_eq']
data.drop(['Λ_m', 'Λ_eq', 'Anion/Cation'], axis = 1, inplace = True)
x = data


print(x)
#print(y)



pca = decomposition.PCA(n_components=3)
pca.fit(x)
x = pca.transform(x)
#print(x)


#x['z'] = pd.Series(map(lambda n: n**(2), x[1]))
#nx = x[0]*x[1]

#print(x)
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

regLasso = linear_model.Lasso(alpha = 0.67) 
regLasso.fit(Xtr, ytr)
pred = regLasso.predict(Xte)
#print(pred)


#METRICS
cvs = []
r = np.linspace(0, 1000)
for a in range(0, 1000):
    regLasso = linear_model.Lasso(alpha = (a+1)/100) 
    cvscore = cross_val_score(regLasso, Xtr, ytr, cv = 4, scoring = 'neg_root_mean_squared_error')
    cvs.append(cvscore.mean())
    
maxScore = max(cvs)
bestAlpha = np.argmax(cvs)
print(maxScore)
print(bestAlpha)
print(cvscore)

#print(cvs)
#plt.scatter(cvs)
#plt.show()
#cvpred = cross_val_predict(Xtr, ytr, cv = 4)

#Take the best of the N models and use it directly
#Take the best of the N models and re-train it on the whole data set
#Keep the N models and rely on the opinion of the majority

#model_pipeline = Pipeline(steps=[('model', RandomForestRegressor(n_estimators=100, random_state=1))])

#cross_validate(regLasso, Xtr, y=ytr,
                #scoring=None, cv=2, 
                #verbose=1,return_train_score=True, return_estimator=False)

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

#plt.cla()
#plt.clf()

#fig = plt.figure(figsize =(5, 5))

#disp = PredictionErrorDisplay(y_true = yTe,y_pred = pred)
disp = PredictionErrorDisplay.from_predictions(y_true = yTe,y_pred = pred, kind="actual_vs_predicted")
disp = PredictionErrorDisplay.from_predictions(y_true = yTe,y_pred = pred, kind="residual_vs_predicted")
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

#plt.show()

