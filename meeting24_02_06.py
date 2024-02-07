#Survey of Matplotlib -> Pyplot
#No new members -> Go on to regression
#Coding

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('MechanicalPropertiesMetals.csv')

htData = data[data['Heat treatment'] != '']

sy = htData['Sy']
ro = htData['Ro']

fig = plt.figure(figsize =(5, 5))

ax = fig.add_subplot(2, 1, 1)
ax.set_xscale('log')
ax.set_xscale('linear')

plt.title('Yield Strength vs Density')
plt.legend('Red stars')
plt.xticks()
plt.yticks()
plt.xlabel('Density')
plt.ylabel('Yield Strength')
plt.ylim((5000,6000))
plt.grid(False)

plt.scatter(sy, ro, marker = '*', color = 'r', s = 1)

ax = fig.add_subplot(2, 1, 2)
ax.set_xscale('log')
ax.set_xscale('linear')

plt.title('Yield Strength vs Density')
plt.legend('Blue dots')
plt.xticks()
plt.yticks()
plt.xlabel('Log Density')
plt.ylabel('Yield Strength')
plt.ylim((2000,5000))
plt.grid(True)

plt.scatter(sy, ro, color = 'blue', marker = '.', s = 2)

plt.show()
