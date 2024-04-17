import pandas as pd

data = pd.read_csv('Abs.csv')

print(data[:, 1])
print(max(data[:, 1]))