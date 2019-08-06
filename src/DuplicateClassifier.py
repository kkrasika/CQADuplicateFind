import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

def train_model(classifier, feature_vector_train, feature_vector_valid,train_y, valid_y, is_neural_net=False):

    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, train_y)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    #print('Predictions : '+ str(predictions.shape))

    return metrics.accuracy_score(predictions, valid_y)

def train_for_data_frame(trainingData):

    print('Row count :' + str(len(trainingData.index)))

    dupYDf = trainingData.loc[trainingData['Dup'] == 'Y']
    print('Dup count :' + str(len(dupYDf.index)))

    x = trainingData[['Q1', 'Q2']]
    y = trainingData['Dup']

    # print('X shape : '+ str(x.shape))
    # print('Y shape : '+ str(y.shape))

    # split the dataset into training and validation datasets
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(x, y)

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(pd.concat((train_x['Q1'],train_x['Q2'])).unique())

    trainq1_trans = count_vect.transform(train_x['Q1'].values)
    trainq2_trans = count_vect.transform(train_x['Q2'].values)

    validq1_trans = count_vect.transform(valid_x['Q1'].values)
    validq2_trans = count_vect.transform(valid_x['Q2'].values)


    xtrain_count =  scipy.sparse.hstack((trainq1_trans,trainq2_trans))
    xvalid_count =  scipy.sparse.hstack((validq1_trans,validq2_trans))

    print('Train X count shape : '+ str(xtrain_count.shape))
    print('Valid X count shape : '+ str(xvalid_count.shape))

    # Naive Bayes on Count Vectors
    nb_model = naive_bayes.MultinomialNB()
    accuracy = train_model(nb_model, xtrain_count, xvalid_count,train_y,valid_y, False)
    print("Accuracy [NB, Count Vectors]: ", accuracy)

    xgb_model = XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0,
                              reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8)
    accuracy = train_model(xgb_model, xtrain_count, xvalid_count,train_y,valid_y, False)
    print("Accuracy [XGB, Count Vectors]: ", accuracy)


def train_for_file(fileName):
    print("----- Training for file: "+fileName+' -----')
    trainDfFromCsv = pd.read_csv('../data/processed/'+fileName+'-training-data.csv', sep=',')
    trainingData = trainDfFromCsv[['Q1', 'Q2', 'Dup']]
    train_for_data_frame(trainingData)

def train_for_list_of_files_each(fileNames):
    for fileName in fileNames:
        train_for_file(fileName)

def train_for_list_of_files_combined(fileNames):
    print("----- Training for combined data: " + str(fileNames)+' -----')
    trainingData = get_df_from_file(fileNames[0])
    i=1
    while i < len(fileNames):
        trainingData = trainingData.append(get_df_from_file(fileNames[i]))
        i += 1

    train_for_data_frame(trainingData)

def get_df_from_file(fileName):
    trainDfFromCsv = pd.read_csv('../data/processed/' + fileName + '-training-data.csv', sep=',')
    trainingData = trainDfFromCsv[['Q1', 'Q2', 'Dup']]
    return trainingData

def main():
    fileNameList = ['android','english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
    #fileNameList = ['webmasters']
    train_for_list_of_files_each(fileNameList)
    train_for_list_of_files_combined(fileNameList)

if __name__ == '__main__':
    main()