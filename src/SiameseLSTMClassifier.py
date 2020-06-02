from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data_for_bert
from config import siamese_config
from tensorflow.python.keras.models import load_model
import numpy as np
from sklearn import preprocessing
from layers.attention import AttentionLayer

from DataSetUtil import get_doc2vec_model_for_csv_file, get_df_from_csv_files_combined, get_train_test_split_of_dataframe, get_df_from_csv_file, get_shuffeled_df_from_csv_files_combined
from CustomLayers import GradientReversal, BertLayer

import tensorflow as tf
# Initialize session
sess = tf.Session()

def evaluate_model(siamese_lstm_model_full, valid_input_ids, valid_input_masks, valid_segment_ids, valid_input2_ids, valid_input2_masks, valid_segment2_ids, valid_labels, valid_domains):

    #preds = list(model.predict(valid_x, verbose=0).ravel())
    score = siamese_lstm_model_full.evaluate([valid_input_ids, valid_input_masks, valid_segment_ids, valid_input2_ids, valid_input2_masks, valid_segment2_ids], [np.array(valid_labels), np.array(valid_domains)], verbose=0)

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

def train_model_for_bert(train_x, train_y, train_domain, sess, filename):

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

    best_model_path = siamese.train_bert_model(train_x, train_y, train_domain, filename, sess, model_save_directory='../data/model/siamese-lstm/')
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

    sentences1 = train_x['Q1']  #+' '+train_x['Q1Ans']
    sentences2 = train_x['Q2']  #+' '+train_x['Q2Ans']
    is_similar = list(train_y)
    train_domain_list = list(train_domain)

    sentences1_validate = valid_x['Q1']  #+' '+valid_x['Q1Ans']
    sentences2_validate = valid_x['Q2']  #+' '+valid_x['Q2Ans']
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

    valid_input_ids, valid_input_masks, valid_segment_ids, valid_input2_ids, valid_input2_masks, valid_segment2_ids, valid_labels, valid_domains = create_test_data_for_bert(sentences_pairs_validate, is_similar_validate, valid_domain_list,
                                                              siamese_config['MAX_SEQUENCE_LENGTH'], sess)

    return sentences_pairs, is_similar, train_domain_list, valid_input_ids, valid_input_masks, valid_segment_ids, valid_input2_ids, valid_input2_masks, valid_segment2_ids, valid_labels, valid_domains

def main():

    fileNameList = ['android','english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
    #fileNameList = ['wordpress']

    # For combined file
    outputFile = open('../data/output/result5.txt', 'a')
    df_combined = get_shuffeled_df_from_csv_files_combined(fileNameList, 5400, 48)
    sentences_pairs, is_similar, train_domain_list, valid_input_ids, valid_input_masks, valid_segment_ids, valid_input2_ids, valid_input2_masks, valid_segment2_ids, valid_labels, valid_domains = get_doc2vec_vectors_train_valid_split(df_combined)
    model_path = train_model_for_bert(sentences_pairs, is_similar, train_domain_list, sess, 'paper/modelv1')
    #model_path = '../data/model/siamese-lstm/paper/' + 'modelv1'+'-'+siamese_config['MODEL_FILE_NAME']
    siamese_lstm_model_full = load_model(model_path, custom_objects={'AttentionLayer' : AttentionLayer, 'GradientReversal' :GradientReversal, 'BertLayer' :BertLayer})
    preds, accuracy = evaluate_model(siamese_lstm_model_full, valid_input_ids, valid_input_masks, valid_segment_ids, valid_input2_ids, valid_input2_masks, valid_segment2_ids, valid_labels, valid_domains)
    print('Accuracy for : ' + 'full' + ' Siamese LSTM 0 : ' + str(accuracy[0]), file=outputFile)
    print('Accuracy for : ' + 'full' + ' Siamese LSTM 1 : ' + str(accuracy[1]), file=outputFile)
    print('Accuracy for : ' + 'full' + ' Siamese LSTM 2 : ' + str(accuracy[2]), file=outputFile)
    print('Accuracy for : ' + 'full' + ' Siamese LSTM 3 : ' + str(accuracy[3]), file=outputFile)
    outputFile.close()

if __name__ == '__main__':
    main()