#Info

#Tasks
#Add people to github
#Music to code
#Data to Analyze

#Concepts

#Coding
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure  
import pandas as pd

data = pd.read_csv('MechanicalPropertiesMetals.csv')

su = data['Su']
sy = data['Sy']
print(su)
print(sy)

#Figure Size
fig = plt.figure(figsize =(4, 4))
ax = fig.add_axes([2, 2, 2, 2])
ax = fig.add_subplot(2, 1, 1)
ax.set_yscale('log')
ax.set_xscale('linear')
plt.scatter(su, sy, color = 'g', marker = 'x', s = 1, label = 'N')
#line = ax.plot(su, color='blue', lw=2)

ax = fig.add_subplot(2, 1, 2)
ax.set_yscale('linear')
ax.set_xscale('linear')
plt.scatter(su, sy, color = 'g', marker = 'x', s = 1, label = 'N')



#plottingData.to_csv('SySu_Data', index = False)

#plt.xticks()
#plt.yticks()

#plt.title("Ultimate Tensile Strength vs Yield Strength") 

plt.show()
