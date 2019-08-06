import pandas as pd
import time
import multiprocessing
import math
import datetime

df = pd.read_json('../data/webmasters_questions.json', orient='index')
df['QuestionID'] = df.index
mydf = df[['QuestionID', 'title', 'dups']]
newdf = pd.DataFrame(columns=['Q1', 'Q2', 'Dup'])
#mydf = mydf[:200]
count = 0
size = mydf['QuestionID'].size
print('Size : '+str(size))
loadForProcessor = math.floor(size/6)
newdf.to_csv('dataset.csv', sep=',')

def basic_func(x,y):
    global size
    global mydf

    for i in range(x, y):
        newdf = pd.DataFrame(columns=['Q1', 'Q2', 'Dup'])
        listOfRecords = []
        q1Id = mydf.iloc[i, 0]
        q1Title = mydf.iloc[i, 1]
        tic = time.process_time()

        for j in range(0, size):
            q2Title = mydf.iloc[j, 1]
            isDup = 'N'
            if q1Id in mydf.iloc[j, 2]:
                isDup = 'Y'
            listOfRecords.append({'Q1': q1Title, 'Q2': q2Title, 'Dup': isDup})

        toc = time.process_time()
        if x == 0:
            print('Iter : ' + str(i)+' took '+str(toc - tic)+' seconds.')

        newdf = newdf.append(listOfRecords, ignore_index=True)
        with open('dataset.csv', 'a') as f:
            newdf.to_csv(f, header=False, sep=',', index=False)

def multiprocessing_func(x):
    newdf = pd.DataFrame(columns=['Q1', 'Q2', 'Dup'])
    a=0
    if x!=0:
        a=loadForProcessor*x+1
    b=loadForProcessor*(x+1)
    basic_func(a,b)

def main():
    startTime = datetime.datetime.now()
    global newdf
    pool = multiprocessing.Pool(processes=6)
    pool.map(multiprocessing_func, range(0, 5))
    pool.close()
    endTime = datetime.datetime.now()
    print('Total time :' + str(endTime-startTime))


if __name__ == '__main__':

    main()
    dfFromCsv = pd.read_csv('dataset.csv', sep=',')
    newdf2 = dfFromCsv[['Q1', 'Q2', 'Dup']]
    print('Row count :' + str(len(newdf2.index)))
    dupYDf = newdf.loc[newdf['Dup'] == 'Y']
    print('Y count :' + str(len(dupYDf.index)))
    print(newdf2.head())