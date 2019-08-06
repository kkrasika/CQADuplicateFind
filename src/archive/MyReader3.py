import pandas as pd

dfFromCsv = pd.read_csv('dataset.csv', sep=',')
newdf2 = dfFromCsv[['Q1', 'Q2', 'Dup']]
print('Row count :' + str(len(newdf2.index)))
dupYDf = newdf2.loc[newdf2['Dup'] == 'Y']
print('Y count :' + str(len(dupYDf.index)))
print(newdf2.head())