import pandas as pd
import scipy
from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
import pandas as pd
from operator import itemgetter
from keras.models import load_model
import numpy as np

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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
    trainingData['Dup'] = trainingData['Dup'].map({'Y': 1, 'N': 0})
    print('Dup count :' + str(len(dupYDf.index)))

    x = trainingData[['Q1', 'Q2']]
    y = trainingData['Dup']


    # split the dataset into training and validation datasets
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(x, y)

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    sentences1 = list(train_x['Q1'])
    sentences2 = list(train_x['Q2'])
    is_similar = list(train_y)

    sentences1_validate = list(valid_x['Q1'])
    sentences2_validate = list(valid_x['Q2'])
    is_similar_validate = list(valid_y)

    tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2,  siamese_config['EMBEDDING_DIM'])

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

    ######## Training ########

    class Configuration(object):
        """Dump stuff here"""

    CONFIG = Configuration()

    CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
    CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
    CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
    CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
    CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
    CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
    CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
    CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']

    siamese = SiameseBiLSTM(CONFIG.embedding_dim, CONFIG.max_sequence_length, CONFIG.number_lstm_units,
                            CONFIG.number_dense_units, CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense,
                            CONFIG.activation_function, CONFIG.validation_split_ratio)

    siamese.train_model(sentences_pairs, is_similar, embedding_meta_data, model_save_directory='../data/model/')
    best_model_path = '../data/model/'+siamese_config['MODEL_FILE_NAME']

    model = load_model(best_model_path)

    test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, sentences_pairs_validate,
                                                              siamese_config['MAX_SEQUENCE_LENGTH'])

    preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
    results = [(x, y, z) for (x, y), z in zip(sentences_pairs_validate, preds)]
    results.sort(key=itemgetter(2), reverse=True)


    score = model.evaluate([test_data_x1, test_data_x2, leaks_test], np.array(is_similar_validate), verbose=0)
    print('Final result :'+ str(score[1]))



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
    #train_for_list_of_files_each(fileNameList)
    train_for_list_of_files_combined(fileNameList)

if __name__ == '__main__':
    main()