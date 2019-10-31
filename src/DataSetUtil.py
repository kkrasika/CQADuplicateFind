import pandas as pd
from sklearn import model_selection
from gensim.models import Doc2Vec

def get_df_from_csv_file(fileName):
    print("----- Data frame for domain : " + str(fileName) + ' -----')
    trainDfFromCsv = pd.read_csv('../data/csv/' + fileName + '-training-data.csv', sep=',')
    trainingData = trainDfFromCsv[['Q1','Q1ID', 'Q2','Q2ID', 'Dup']]
    return trainingData

def get_doc2vec_model_for_csv_file(fileName):
    try:
        doc2vec_model = Doc2Vec.load('../data/model/doc2vec/int-labeled/' + str(fileName) + "-d2v.model")
    except FileNotFoundError as e:
        print(e.errno)

    return doc2vec_model

def get_df_from_csv_files_combined(fileNames):
    print("----- Data frame for combined data : " + str(fileNames)+' -----')
    trainingData = get_df_from_csv_file(fileNames[0])
    i=1
    while i < len(fileNames):
        trainingData = trainingData.append(get_df_from_csv_file(fileNames[i]))
        i += 1
    return trainingData

# returns train_x, valid_x, train_y, valid_y
def get_train_test_split_of_dataframe(dataFrame, withQid):
    print('Row count :' + str(len(dataFrame.index)))

    dupYDf = dataFrame.loc[dataFrame['Dup'] == 1]
    print('Dup count :' + str(len(dupYDf.index)))

    x = dataFrame[['Q1', 'Q2']]
    y = dataFrame['Dup']

    if withQid:
        x = dataFrame[['Q1','Q1ID', 'Q2','Q2ID']]

    return model_selection.train_test_split(x, y)