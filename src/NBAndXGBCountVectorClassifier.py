import pandas as pd
import scipy
from xgboost import XGBClassifier
from sklearn import preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import CountVectorizer
from DataSetUtil import get_df_from_csv_file, get_df_from_csv_files_combined, get_train_test_split_of_dataframe

def evaluate_model(classifier, feature_vector_valid, valid_y):
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions, valid_y)

def train_model(train_x, train_y, model_type):

    classifier = None
    if model_type is 'NB':
        classifier = naive_bayes.MultinomialNB()
    elif model_type is 'XGB':
        classifier = XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0,
                                  reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8)

    # fit the training dataset on the classifier
    return classifier.fit(train_x, train_y)

def get_count_vectors_train_valid_split(df_for_file):
    train_x, valid_x, train_y, valid_y = get_train_test_split_of_dataframe(df_for_file, False)
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

    return xtrain_count, xvalid_count, train_y, valid_y

def main():
    fileNameList = ['android','english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']

    # Split / Model / Evaluate for each data set and for combined.

    # For each file
    for fileName in fileNameList:
        df_for_file = get_df_from_csv_file(fileName)
        train_x, valid_x, train_y, valid_y = get_count_vectors_train_valid_split(df_for_file)

        model_nb = train_model(train_x, train_y, 'NB')
        accuracy = evaluate_model(model_nb, valid_x, valid_y)
        print('Accuracy for : '+fileName+' NB ' + str(accuracy))
        model_xgb = train_model(train_x, train_y, 'XGB')
        accuracy = evaluate_model(model_xgb, valid_x, valid_y)
        print('Accuracy for : '+fileName+' XGB ' + str(accuracy))

    # For combined file
    df_combined = get_df_from_csv_files_combined(fileNameList)
    train_x, valid_x, train_y, valid_y = get_count_vectors_train_valid_split(df_combined)

    model_nb = train_model(train_x, train_y, 'NB')
    accuracy = evaluate_model(model_nb, valid_x, valid_y)
    print('Accuracy for : ' + 'Full' + ' NB ' + str(accuracy))
    model_xgb = train_model(train_x, train_y, 'XGB')
    accuracy = evaluate_model(model_xgb, valid_x, valid_y)
    print('Accuracy for : ' + 'Full' + ' XGB ' + str(accuracy))

if __name__ == '__main__':
    main()