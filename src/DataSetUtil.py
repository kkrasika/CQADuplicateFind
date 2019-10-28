import pandas as pd
from sklearn import model_selection

def get_df_from_csv_file(fileName):
    print("----- Data frame for domain : " + str(fileName) + ' -----')
    trainDfFromCsv = pd.read_csv('../data/csv/' + fileName + '-training-data.csv', sep=',')
    trainingData = trainDfFromCsv[['Q1','Q1ID', 'Q2','Q2ID', 'Dup']]
    return trainingData

def get_df_from_csv_files_combined(fileNames):
    print("----- Data frame for combined data : " + str(fileNames)+' -----')
    trainingData = get_df_from_csv_file(fileNames[0])
    i=1
    while i < len(fileNames):
        trainingData = trainingData.append(get_df_from_csv_file(fileNames[i]))
        i += 1
    return trainingData

# returns train_x, valid_x, train_y, valid_y
def get_train_test_split_of_dataframe(dataFrame):
    print('Row count :' + str(len(dataFrame.index)))

    dupYDf = dataFrame.loc[dataFrame['Dup'] == 1]
    print('Dup count :' + str(len(dupYDf.index)))

    x = dataFrame[['Q1', 'Q2']]
    y = dataFrame['Dup']

    return model_selection.train_test_split(x, y)