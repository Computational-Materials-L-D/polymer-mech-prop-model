# rho0 = 1.0051g/cm^3 
# rhof = 0.9714g/cm^3 => 0.9714275906682729
# Equilibrium time ~= 30ns-50ns
#

import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

theta = np.loadtxt('theta.txt')
rho = np.loadtxt('rho.txt')
Et = np.loadtxt('Et.txt')
Ep = np.loadtxt('Ep.txt')
time = np.linspace(0, len(Et)+1, 1)
print(time)

# time = Et[1:, 0]
# Et = Et[1:, 1]
# dEt = Et[1:]-Et[:-1]

lrpEt = linregress(Et[2500:, 0], Et[2500:, 1])
aEt = lrpEt[0]
bEt = lrpEt[1]
lbfEt = aEt*Et[2500:, 0] + bEt
print(aEt)
print(bEt)
# lrpdE = linregress(time[1:], dEt)
# adE = lrpdE[0]
# bdE = lrpdE[1]
# lbfdE = lrpdE[0]*time + lrpdE[1]
# print(adE)
# print(bdE)

fig = plt.figure(figsize =(5, 5))
fig.tight_layout()

fig.add_subplot(2, 2, 1)

# plt.title('Total Energy vs Time')
plt.xticks()
plt.yticks()
plt.xlabel('time (ps)')
plt.ylabel('Etotal (kcal/mol)')
plt.plot(Et[:, 0], Et[:, 1], color = 'blue', linestyle = 'solid', linewidth = 0.1)
print(np.mean(Et[3000:5000, 1]))
print(np.median(Et[3000:5000, 1]))
#plt.scatter(Et[2500:, 0], lbfEt, color = 'orange', marker = '*', s = 0.01)

fig.add_subplot(2, 2, 2)

# plt.title('Potential Energy vs Time')
plt.xticks()
plt.yticks()
plt.xlabel('time (ps)')
plt.ylabel('Epotential (kcal/mol)')
plt.scatter(Ep[:, 0], Ep[:, 1], color = 'blue', marker = '.', s = 0.5)
print(np.mean(Ep[3000:5000, 1]))
print(np.median(Ep[3000:5000, 1]))

fig.add_subplot(2, 2, 3)

# plt.title('Density vs Time')
plt.xticks()
plt.yticks()
plt.xlabel('time (ps)')
plt.ylabel('Density (g/cm^-3)')
plt.scatter(rho[:, 0], rho[:, 1], color = 'blue', marker = '.', s = 0.5)

print(np.mean(rho[3000:5000, 1]))
print(np.median(rho[3000:5000, 1]))

ax = fig.add_subplot(2, 2, 4)

# plt.title('Temperature vs Time')
plt.xticks()
plt.yticks()
plt.xlabel('time (ps)')
plt.ylabel('Temperature (K)')
plt.scatter(theta[:, 0], theta[:, 1], color = 'blue', marker = '.', s = 0.5)
print(np.mean(theta[3000:5000, 1]))
print(np.median(theta[3000:5000, 1]))


# fig.add_subplot(4, 1, 2)

# plt.title('Derivative Total Energy vs Time')
# plt.xticks()
# plt.yticks()
# plt.xlabel('time (ns)')
# plt.ylabel('dEtotal (kcal/mol)')
# plt.scatter(time[1:], dEt, color = 'green', marker = '.', s = 0.3)
# plt.scatter(time[1:], lbfdE[1:], color = 'red', marker = '*', s = 0.05)



#ax.set_xscale('linear')
#ax.set_xscale('linear')
#plt.grid(True)


plt.show()