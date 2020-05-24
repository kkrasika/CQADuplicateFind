from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
from tensorflow.python.keras.models import load_model
import numpy as np
from sklearn import preprocessing
from layers.attention import AttentionLayer

from DataSetUtil import get_doc2vec_model_for_csv_file, get_df_from_csv_files_combined, get_train_test_split_of_dataframe, get_df_from_csv_file, get_shuffeled_df_from_csv_files_combined
from model import GradientReversal

def evaluate_model(model, valid_x, valid_y, valid_domain_list):

    #preds = list(model.predict(valid_x, verbose=0).ravel())
    score = model.evaluate(valid_x, [np.array(valid_y), np.array(valid_domain_list)], verbose=0)

    return '', score

def train_model(train_x, train_y, train_domain, embedding_meta_data, filename):

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

    best_model_path = siamese.train_model(train_x, train_y, train_domain, embedding_meta_data,filename, model_save_directory='../data/model/siamese-lstm/')
    return best_model_path

def get_doc2vec_vectors_train_valid_split(trainingData):

    # split the dataset into training and validation datasets
    train_x, valid_x, train_y_orig, valid_y_orig = get_train_test_split_of_dataframe(trainingData, False)

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y_orig['Dup'])
    valid_y = encoder.fit_transform(valid_y_orig['Dup'])
    train_domain = encoder.fit_transform(train_y_orig['DomainId'])
    valid_domain = encoder.fit_transform(valid_y_orig['DomainId'])

    sentences1 = train_x['Q1']
    sentences2 = train_x['Q2']  +' '+train_x['Q2Ans']
    is_similar = list(train_y)
    train_domain_list = list(train_domain)

    sentences1_validate = valid_x['Q1']
    sentences2_validate = valid_x['Q2']  +' '+train_x['Q2Ans']
    is_similar_validate = list(valid_y)
    valid_domain_list = list(valid_domain)

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
    test_data_x = [test_data_x1, test_data_x2, leaks_test]

    return sentences_pairs, is_similar, train_domain_list, test_data_x, is_similar_validate, valid_domain_list, embedding_meta_data

def main():

    fileNameList = ['android','english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
    #fileNameList = ['english', 'gaming', 'physics', 'tex','unix']
    #fileNameList = ['mathematica']
    #fileNameList = ['physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
    #fileNameList = ['webmasters']
    #fileNameList = ['android', 'english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters']
    # Split / Model / Evaluate for each data set and for combined.

    # For combined file
    outputFile = open('../data/output/result5.txt', 'a')
    df_combined = get_shuffeled_df_from_csv_files_combined(fileNameList, 5400, 48)
    train_x, train_y, train_domain_list, valid_x, valid_y, valid_domain_list, embedding_meta_data = get_doc2vec_vectors_train_valid_split(df_combined)
    model_path = train_model(train_x, train_y, train_domain_list, embedding_meta_data, 'pv-classifier-v1')
    #model_path = '../data/model/siamese-lstm/' + 'full-q2_with_answers-'+'-'+siamese_config['MODEL_FILE_NAME']
    siamese_lstm_model_full = load_model(model_path, custom_objects={'AttentionLayer' : AttentionLayer, 'GradientReversal' :GradientReversal})
    preds, accuracy = evaluate_model(siamese_lstm_model_full, valid_x, valid_y, valid_domain_list)
    print('Accuracy for : ' + 'full' + ' Siamese LSTM 0 : ' + str(accuracy[0]), file=outputFile)
    print('Accuracy for : ' + 'full' + ' Siamese LSTM 1 : ' + str(accuracy[1]), file=outputFile)
    print('Accuracy for : ' + 'full' + ' Siamese LSTM 2 : ' + str(accuracy[2]), file=outputFile)
    print('Accuracy for : ' + 'full' + ' Siamese LSTM 3 : ' + str(accuracy[3]), file=outputFile)
    outputFile.close()

    # For each file
    '''
    for fileName in fileNameList:
        outputFile = open('../data/output/result.txt', 'a')
        df_for_file = get_df_from_csv_file(fileName)
        train_x, train_y, train_domain_list, valid_x, valid_y, valid_domain_list, embedding_meta_data = get_doc2vec_vectors_train_valid_split(df_for_file)

        preds, accuracy = evaluate_model(siamese_lstm_model_full, valid_x, valid_y, valid_domain_list)
        print('Accuracy for : '+fileName+' Siamese LSTM ' + str(str(accuracy[3])), file=outputFile)

        model_path_domain = train_model(train_x, train_y, train_domain_list, embedding_meta_data, fileName)
        siamese_lstm_model_domain = load_model(model_path_domain, custom_objects={'AttentionLayer': AttentionLayer, 'GradientReversal': GradientReversal})
        preds, accuracy = evaluate_model(siamese_lstm_model_domain, valid_x, valid_y, valid_domain_list)
        print('Accuracy [Domain] for : ' + fileName + ' Siamese LSTM ' + str(str(accuracy[3])), file=outputFile)

        outputFile.close()
    '''

if __name__ == '__main__':
    main()