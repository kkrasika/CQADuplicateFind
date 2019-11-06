# This converts json data sets in to csv files which maintains given number of question pairs including duplicates of defined percentage

import pandas as pd
import math

recordCount = 1000
duplicatePercentage = 33

def get_records(df, recordids):
    return df.loc[df['QuestionID'].isin(recordids)]

def check_dup(q1, q2):

    q1Dupids = q1.loc['dups']
    q2Dupids = q2.loc['dups']

    return q1.loc['QuestionID'] in q2Dupids and q2.loc['QuestionID'] in q1Dupids

def prepareDataSet(df, inputData):

    count = 0
    dupCount = 0
    maxNonDupCountPerQuestion = (math.floor(100/duplicatePercentage))-1

    result = []

    while count < recordCount:
        dupRow = inputData[dupCount]
        q1Title = dupRow.loc['title']
        q1Id = str(dupRow.loc['QuestionID'])
        dupids = dupRow.loc['dups']
        dups = get_records(df, dupids)
        result.append({'Q1ID': q1Id, 'Q1': q1Title, 'Q2ID': str(dups.iloc[0].loc['QuestionID']), 'Q2': dups.iloc[0].loc['title'], 'Dup': 1})
        count += 1
        dupCount += 1

        nonDupCountPerQuestion = 0
        while nonDupCountPerQuestion < maxNonDupCountPerQuestion:
            if not check_dup(dupRow, df.iloc[count]):
                q2Title = df.iloc[count].loc['title']
                q2Id  = str(df.iloc[count].loc['QuestionID'])
                result.append({'Q1ID': q1Id, 'Q1': q1Title,'Q2ID': q2Id,  'Q2': q2Title, 'Dup': 0})
                nonDupCountPerQuestion += 1
                count += 1

        print('Progressing : '+ str(count)+' / '+ str(recordCount))

    return result

def create_csv_for_file(filename):

    print('Training file generation starts : ' + str(filename))

    df = pd.read_json('../data/json/'+filename+'_questions.json', orient='index')
    df['QuestionID'] = df.index
    df = df[['QuestionID', 'title', 'dups']]
    size = df['QuestionID'].size

    dup_rows = []

    for i in range(0, size):
        if df.iloc[i].loc['dups']:
            dup_rows.append(df.iloc[i])

    trainingRecords = prepareDataSet(df, dup_rows)

    trainDf = pd.DataFrame(columns=['Q1', 'Q2', 'Dup'])

    trainDf = trainDf.append(trainingRecords, ignore_index=True, sort=True)

    trainDf.to_csv('../data/csv/'+str(filename)+'-training-data.csv', sep=',')

    print('Training file generation completed : ' + str(filename))

def create_csv_for_files(fileNames):
    for fileName in fileNames:
        create_csv_for_file(fileName)

def main():
    fileNameList = ['android','english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
    #fileNameList = ['webmasters']
    create_csv_for_files(fileNameList)

if __name__ == '__main__':
    main()
