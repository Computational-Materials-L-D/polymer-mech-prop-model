from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

data = genfromtxt('dp.txt', delimiter=' ')
#print(data)

x = np.asarray(data[:, 0])
y = np.asarray(data[:, 1])
rho = 0.7209
D = 120

#print(x)
#print(y)

def hpFit(z, z50, w):
    den = 1/2*rho*(1 + np.tanh((z- z50)/(0.5*w)))
    return den

parameters, cov = curve_fit(hpFit, x, y)
fit_z50 = parameters[0] 
fit_w = parameters[1]
print(cov)

d = D -2*fit_z50

print(fit_z50)
print(fit_w)
print(d)

plt.scatter(x, y, marker = 'o', s = 10, label = 'density')
plt.plot([25, 25], [0, max(y)], linestyle = '-', color = 'r', linewidth = 0.1)
plt.plot([0, 50], [max(y)*0.8, max(y)*0.8], linestyle = '-', color = 'r', linewidth = 0.1)
plt.plot([0, 50], [max(y)*0.5, max(y)*0.5], linestyle = '-', color = 'r', linewidth = 0.1)
plt.plot([0, 50], [max(y)*0.2, max(y)*0.2], linestyle = '-', color = 'r', linewidth = 0.1)
plt.show()