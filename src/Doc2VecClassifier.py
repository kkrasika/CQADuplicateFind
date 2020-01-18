import pandas as pd
from xgboost import XGBClassifier as xgb

import DataSetUtil as dsu

from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config

from DataSetUtil import get_doc2vec_model_for_csv_file, get_df_from_csv_files_combined, get_train_test_split_of_dataframe, get_df_from_csv_file, get_limited_df_from_csv_file

from inputHandler import create_train_dev_set
import numpy as np
import gc


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

def get_doc2vec_vectors_train_valid_split(trainingData):

    # split the dataset into training and validation datasets
    train_x, valid_x, train_y, valid_y = get_train_test_split_of_dataframe(trainingData, False)

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    sentences1 = train_x['Q1']
    sentences2 = train_x['Q2']
    is_similar = list(train_y)

    sentences1_validate = valid_x['Q1']
    sentences2_validate = valid_x['Q2']
    is_similar_validate = list(valid_y)

    tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2, siamese_config['EMBEDDING_DIM'])

    embedding_meta_data = {
        'tokenizer': tokenizer,
        'embedding_matrix': embedding_matrix
    }

    ## creating sentence pairs
    sentences_pairs = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
    del sentences1
    del sentences2

    sentences_pairs_validate = [(x1, x2) for x1, x2 in zip(sentences1_validate, sentences2_validate)]
    del sentences1_validate
    del sentences2_validate

    test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, sentences_pairs_validate,
                                                              siamese_config['MAX_SEQUENCE_LENGTH'])
    test_data_x = [test_data_x1, test_data_x2]

    return sentences_pairs, is_similar, test_data_x, is_similar_validate, embedding_meta_data

def f(x):
    size_value = 800
    return {
        'android' : size_value, 'english' : size_value, 'gaming' : size_value, 'gis' : size_value, 'mathematica' : size_value,
        'physics' : size_value, 'programmers' : size_value, 'stats' : size_value, 'tex' : size_value, 'unix' : size_value,
        'webmasters' : size_value #, 'wordpress' : 2000,
    }[x]

def main():
    fileNameList = ['android','english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']

    # Split / Model / Evaluate for each data set and for combined.
    # For each file
    for fileName in fileNameList:
        #df_for_file = get_df_from_csv_file(fileName)
        #doc2vec_model = get_doc2vec_model_for_csv_file(fileName)
        #train_x, valid_x, train_y, valid_y = get_doc2vec_vectors_train_valid_split(df_for_file, doc2vec_model)

        df_for_file = get_limited_df_from_csv_file(fileName, f(fileName))
        train_x, train_y, valid_x, valid_y, embedding_meta_data = get_doc2vec_vectors_train_valid_split(df_for_file)

        tokenizer, embedding_matrix = embedding_meta_data['tokenizer'], embedding_meta_data['embedding_matrix']

        train_data_x1, train_data_x2, train_labels, leaks_train, val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, train_x, train_y, 80, 0.1)

        sentences_pairs = [(x1, x2) for x1, x2 in zip(train_data_x1, train_data_x2)]
        sentences_pairs_validate = [(x1, x2) for x1, x2 in zip(val_data_x1, val_data_x2)]

        print('test')

        del train_x, train_y, valid_x, valid_y, embedding_meta_data
        del tokenizer, embedding_matrix
        del train_data_x1, train_data_x2, leaks_train, val_data_x1, val_data_x2, leaks_val
        gc.collect()

        train_pairs_array = np.asarray(sentences_pairs)
        valid_pairs_array = np.asarray(sentences_pairs_validate)

        del sentences_pairs, sentences_pairs_validate
        gc.collect()

        reshaped_train_array = np.reshape(train_pairs_array, (540,122880))
        reshaped_valid_array = np.reshape(valid_pairs_array, (60, 122880))

        del train_pairs_array, valid_pairs_array
        gc.collect()

        xx1 = csr_matrix(reshaped_train_array)
        xx2 = csr_matrix(reshaped_valid_array)

        del reshaped_train_array, reshaped_valid_array
        gc.collect()

        model_xgb = train_model(xx1, train_labels, 'XGB')
        accuracy = evaluate_model(model_xgb, xx2, val_labels)
        outputFile = open('../data/output/result.txt', 'a')
        print('Accuracy for : '+fileName+' XGB ' + str(accuracy), file=outputFile)
        outputFile.close()

        del df_for_file
        del train_labels, val_labels
        del xx1
        del xx2
        del model_xgb
        gc.collect()

    # For combined file
    df_combined = get_df_from_csv_files_combined(fileNameList)
    doc2vec_model = get_doc2vec_model_for_csv_file('full')
    train_x, valid_x, train_y, valid_y = get_doc2vec_vectors_train_valid_split(df_combined, doc2vec_model)

    model_xgb = train_model(train_x, train_y, 'XGB')
    accuracy = evaluate_model(model_xgb, valid_x, valid_y)
    print('Accuracy for : ' + 'Full' + ' XGB ' + str(accuracy))

if __name__ == '__main__':
    main()