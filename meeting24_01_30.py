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
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('MechanicalPropertiesMetals.csv')
plottingData = data.drop(['Heat treatment','G','ID','Material','Std','A5','Bhn','mu','pH','Desc','HV'], axis = 1, inplace = True) 
su = plottingData['Su']
sy = plottingData['Sy']
print(su)
print(sy)

#plottingData.to_csv('SySu_Data', index = False)

plt.scatter(su, sy, color = 'b')
plt.xticks()
plt.yticks()


plt.plot([1, 1000], [1, 1000], '--')
