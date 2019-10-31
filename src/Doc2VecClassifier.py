import pandas as pd
from xgboost import XGBClassifier as xgb

from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

from DataSetUtil import get_doc2vec_model_for_csv_file, get_df_from_csv_files_combined, get_train_test_split_of_dataframe, get_df_from_csv_file

def evaluate_model(classifier, feature_vector_valid, valid_y):
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions, valid_y)

def train_model(train_x, train_y, model_type):
    model = None
    if model_type is 'XGB':
        model = xgb(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0,
                                  reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8)
        model.fit(train_x, train_y)
    return model

def get_doc2vec_vectors_train_valid_split(df_for_file, doc2vec_model):

    train_x, valid_x, train_y, valid_y = get_train_test_split_of_dataframe(df_for_file, True)

    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(pd.concat((train_x['Q1'], train_x['Q2'])).unique())

    xtrain_count_vector = []
    xvalid_count_vector = []

    for qid in train_x['Q1ID'].values:
        try:
            xtrain_count_vector.append(doc2vec_model.docvecs.vectors_docs[qid])
        except KeyError as e:
            print(e)

    for qid in valid_x['Q1ID'].values:
        try:
            xvalid_count_vector.append(doc2vec_model.docvecs.vectors_docs[qid])
        except KeyError as e:
            print(e)

    print("Train size : " + str(len(train_y)))
    print('Dup count :' + str(sum(train_y)))
    print("Valid size : " + str(len(valid_y)))
    print('Dup count :' + str(sum(valid_y)))

    xgtrain = csr_matrix(xtrain_count_vector)
    xgvalid = csr_matrix(xvalid_count_vector)

    return xgtrain, xgvalid, train_y, valid_y

def main():
    fileNameList = ['android','english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']

    # Split / Model / Evaluate for each data set and for combined.
    # For each file
    for fileName in fileNameList:
        df_for_file = get_df_from_csv_file(fileName)
        doc2vec_model = get_doc2vec_model_for_csv_file(fileName)
        train_x, valid_x, train_y, valid_y = get_doc2vec_vectors_train_valid_split(df_for_file, doc2vec_model)

        model_xgb = train_model(train_x, train_y, 'XGB')
        accuracy = evaluate_model(model_xgb, valid_x, valid_y)
        print('Accuracy for : '+fileName+' XGB ' + str(accuracy))

    # For combined file
    df_combined = get_df_from_csv_files_combined(fileNameList)
    doc2vec_model = get_doc2vec_model_for_csv_file('full')
    train_x, valid_x, train_y, valid_y = get_doc2vec_vectors_train_valid_split(df_combined, doc2vec_model)

    model_xgb = train_model(train_x, train_y, 'XGB')
    accuracy = evaluate_model(model_xgb, valid_x, valid_y)
    print('Accuracy for : ' + 'Full' + ' XGB ' + str(accuracy))

if __name__ == '__main__':
    main()