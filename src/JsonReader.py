import pandas as pd
import time
import multiprocessing
import math
import datetime

maxTrain = 223
testToTrainRatio = 0.1
duplicatePercentage = 0.05

def get_records(df, recordids):
    return df.loc[df['QuestionID'].isin(recordids)]

def check_dup(q1, q2):

    q1Dupids = q1.loc['dups']
    q2Dupids = q2.loc['dups']

    return q1.loc['QuestionID'] in q2Dupids and q2.loc['QuestionID'] in q1Dupids

def prepareDataSet(df, isTrain, inputData):

    count = 0
    startingIndex = 0
    endingIndex = maxTrain
    maxCount = maxTrain
    processLabel = 'Training'
    pivot = math.floor(maxTrain*(1-testToTrainRatio))
    maxNonDupCount = math.floor(duplicatePercentage*100)-1

    if isTrain:
        endingIndex = pivot
    else:#
        processLabel = 'Testing'
        startingIndex =  pivot+1+maxNonDupCount
        endingIndex = endingIndex + maxNonDupCount
        maxCount = maxTrain*testToTrainRatio

    result = []

    for k in range(startingIndex, endingIndex):

        dupRow = inputData[k]
        q1Title = dupRow.loc['title']
        q1Id = str(dupRow.loc['QuestionID'])
        dupids = dupRow.loc['dups']
        dups = get_records(df, dupids)
        result.append({'Q1ID': q1Id, 'Q1': q1Title, 'Q2ID': str(dups.iloc[0].loc['QuestionID']), 'Q2': dups.iloc[0].loc['title'], 'Dup': 'Y'})
        nonDupCount = 0

        # if (endingIndex-k-1) < math.floor(maxCount*duplicatePercentage):
        #     break

        for p in range(k + 1, k+1+maxNonDupCount):

            #print(processLabel + ' in progress ... K = '+str(k)+' P = '+str(p))

            if not check_dup(dupRow, inputData[p]):
                q2Title = inputData[p].loc['title']
                q2Id  = str(inputData[p].loc['QuestionID'])
                result.append({'Q1ID': q1Id, 'Q1': q1Title,'Q2ID': q2Id,  'Q2': q2Title, 'Dup': 'N'})

    return result

def create_csv_for_file(filename):

    df = pd.read_json('../data/raw/'+filename+'_questions.json', orient='index')
    df['QuestionID'] = df.index
    df = df[['QuestionID', 'title', 'dups']]
    size = df['QuestionID'].size

    #dup_rows = pd.DataFrame(columns=['Q1', 'Q2'])
    dup_rows = []
    #non_dup_rows = pd.DataFrame(columns=['Q1', 'Q2', 'Dup'])
    no_dup_rows = []
    for i in range(0, size):
        qid = df.iloc[i].loc['QuestionID']
        dupids = df.iloc[i].loc['dups']

        if dupids:
            dup_rows.append(df.iloc[i])
        else :
            no_dup_rows.append(df.iloc[i])

    #print('Dup count : ' + str(len(dup_rows)))
    #print('Non Dup Count : ' + str(len(no_dup_rows)))

    trainingRecords = prepareDataSet(df, True, dup_rows)
    #testingRecords = prepareDataSet(df,False, dup_rows)

    #print('Train Set : '+str(len(trainingRecords)))
    #print('Test Set : '+str(len(testingRecords)))

    trainDf = pd.DataFrame(columns=['Q1', 'Q2', 'Dup'])
    #testDf = pd.DataFrame(columns=['Q1', 'Q2', 'Dup'])

    trainDf = trainDf.append(trainingRecords, ignore_index=True)
    #testDf = testDf.append(testingRecords, ignore_index=True)

    trainDf.to_csv('../data/processed/'+str(filename)+'-training-data.csv', sep=',')
    #testDf.to_csv('testing-data.csv', sep=',')

    print('Training file generated for : ' + str(filename))

def create_csv_for_files(fileNames):
    for fileName in fileNames:
        create_csv_for_file(fileName)

def main():
    #fileNameList = ['a ndroid','english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
    fileNameList = ['webmasters', 'wordpress']
    create_csv_for_files(fileNameList)

if __name__ == '__main__':
    main()
