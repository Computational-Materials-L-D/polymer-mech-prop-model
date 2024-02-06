#Info

#Tasks
#Add people to github
#Music to code
#Data to Analyze

#Concepts

#Coding
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('MechanicalPropertiesMetals.csv')

#su = data['Su']
#sy = data['Sy']
#e = data['E']
#g = data['G']
#ro = data['Ro']
#ann = data['Heat Treatment']

annData = data[data['Heat treatment'] != '']
annE = annData['E']
annRo = annData['Ro']

#print(su)
#print(sy)

#Figure Size
# fig = plt.figure(figsize =(4, 4))
# ax = fig.add_axes([2, 2, 2, 2])
# ax = fig.add_subplot(2, 1, 1)
# ax.set_yscale('log')
# ax.set_xscale('linear')
# plt.scatter(su, sy, color = 'g', marker = 'x', s = 1, label = 'N')
#line = ax.plot(su, color='blue', lw=2)

# ax = fig.add_subplot(2, 2, 1)
# ax.set_yscale('linear')
# ax.set_xscale('linear')
# plt.scatter(su, sy, color = 'g', marker = 'x', s = 1, label = 'N')


fig = plt.figure(figsize =(3, 3))
ax = fig.add_subplot(2, 1, 1)
ax.set_xscale('linear')
ax.set_xscale('linear')
plt.scatter(annRo, annE, color = 'r', marker = '.', label = '')
plt.title("Young Modulus vs Density") 
plt.legend('Points Red')

ax = fig.add_subplot(2, 1, 2)
ax.set_xscale('log')
ax.set_xscale('linear')
plt.scatter(annRo, annE, color = 'b', marker = '*', label = '')
plt.title("Log Young Modulus vs Density") 
plt.legend('Stars Blue')

#plottingData.to_csv('SySu_Data', index = False)

plt.xticks()
plt.yticks()
plt.show()
