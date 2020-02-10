# keras imports
from tensorflow.python.keras.layers import Dense, Input, LSTM, Dropout, Bidirectional, Reshape, Concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model
from config import siamese_config

# std imports
import time
import gc
import os
from tensorflow.python.keras.engine import Layer

from inputHandler import create_train_dev_set
from layers.attention import AttentionLayer
from keras.layers.wrappers import TimeDistributed
import tensorflow.python.keras.backend as K
from tensorflow.python.framework import ops
import tensorflow as tf
from math import exp


class SiameseBiLSTM:
    def __init__(self, embedding_dim, max_sequence_length, number_lstm, number_dense, rate_drop_lstm, 
                 rate_drop_dense, hidden_activation, validation_split_ratio):
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.number_lstm_units = number_lstm
        self.rate_drop_lstm = rate_drop_lstm
        self.number_dense_units = number_dense
        self.activation_function = hidden_activation
        self.rate_drop_dense = rate_drop_dense
        self.validation_split_ratio = validation_split_ratio
        self.lambda_reversal = 0.25

    def train_model(self, sentences_pair, is_similar, train_domain, embedding_meta_data, filename, model_save_directory='./'):
        """
        Train Siamese network to find similarity between sentences in `sentences_pair`
            Steps Involved:
                1. Pass the each from sentences_pairs  to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encodes and passed to dense layer.
                3. Pass the  dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            sentences_pair (list): list of tuple of sentence pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix
            model_save_directory (str): working directory for where to save models

        Returns:
            return (best_model_path):  path of best model
        """
        tokenizer, embedding_matrix = embedding_meta_data['tokenizer'], embedding_meta_data['embedding_matrix']

        train_data_x1, train_data_x2, train_labels, domains_train, leaks_train, \
        val_data_x1, val_data_x2, val_labels, domains_val, leaks_val = create_train_dev_set(tokenizer, sentences_pair,
                                                                               is_similar, train_domain, self.max_sequence_length,
                                                                               self.validation_split_ratio)

        if train_data_x1 is None:
            print("++++ !! Failure: Unable to train model ++++")
            return None

        nb_words = len(tokenizer.word_index) + 1

        # Creating word embedding layer
        embedding_layer = Embedding(nb_words, self.embedding_dim, weights=[embedding_matrix],
                                    input_length=self.max_sequence_length, trainable=False)

        # Creating LSTM Encoder
        lstm_layer_encode = Bidirectional(LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm, return_sequences=True))

        # Creating LSTM Encoder layer for First Sentence
        sequence_1_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1_out = lstm_layer_encode(embedded_sequences_1)

        # Creating LSTM Dncoder
        lstm_layer_decode = Bidirectional(LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm, return_sequences=True))

        # Creating LSTM Encoder layer for Second Sentence
        sequence_2_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        x2_out = lstm_layer_decode(embedded_sequences_2)

        # Attention layer
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([x1_out, x2_out])

        # Concat attention input and decoder LSTM output
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([x2_out, attn_out])

        # Creating LSTM Dncoder
        lstm_layer_final = Bidirectional(LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm))
        lstm_out = lstm_layer_final(decoder_concat_input)

        # Domain Adaptation section - classify the domain
        flip_layer = GradientReversal(self.lambda_reversal)
        dann_in = flip_layer(lstm_out)
        dnn_out = Dense(11, activation='softmax', name='domain_classifier')(dann_in)

        # Creating leaks input
        leaks_input = Input(shape=(leaks_train.shape[1],))
        #leaks_embeddings = embedding_layer(leaks_input)
        leaks_dense = Dense(int(self.number_dense_units/2), activation=self.activation_function)(leaks_input)

        # Merging two LSTM encodes vectors from sentences to
        # pass it to dense layer applying dropout and batch normalisation
        merged = concatenate([lstm_out, leaks_dense], axis=1)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        merged = Dense(self.number_dense_units, activation=self.activation_function)(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)

        preds = Dense(1, activation='sigmoid', name='duplicate_classifier')(merged)

        model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], outputs=[preds, dnn_out])

        # model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
        model.compile(loss={'duplicate_classifier': 'binary_crossentropy', 'domain_classifier': 'sparse_categorical_crossentropy'}, optimizer='nadam', metrics=['acc'], loss_weights={'duplicate_classifier': 1, 'domain_classifier': 1})
        #model.compile(loss={'duplicate_classifier': 'binary_crossentropy', 'domain_classifier': 'sparse_categorical_crossentropy'}, optimizer='nadam', metrics=['acc'])

        # early_stopping = EarlyStopping(monitor='val_loss', patience=50)
        early_stopping = EarlyStopping(monitor='val_duplicate_classifier_acc', min_delta=0.001, patience=5)

        checkpoint_dir = model_save_directory

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + filename+'-'+siamese_config['MODEL_FILE_NAME']

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

        print(model.summary())

        model.fit([train_data_x1, train_data_x2, leaks_train], [train_labels, domains_train],
                  validation_data=([val_data_x1, val_data_x2, leaks_val], [val_labels, domains_val]),
                  epochs=50, batch_size=100, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard],verbose=1)

        return bst_model_path


    def update_model(self, saved_model_path, new_sentences_pair, is_similar, embedding_meta_data):
        """
        Update trained siamese model for given new sentences pairs 
            Steps Involved:
                1. Pass the each from sentences from new_sentences_pair to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encodes and passed to dense layer.
                3. Pass the  dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            model_path (str): model path of already trained siamese model
            new_sentences_pair (list): list of tuple of new sentences pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix

        Returns:
            return (best_model_path):  path of best model
        """
        tokenizer = embedding_meta_data['tokenizer']
        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, new_sentences_pair,
                                                                               is_similar, self.max_sequence_length,
                                                                               self.validation_split_ratio)
        model = load_model(saved_model_path)
        model_file_name = saved_model_path.split('/')[-1]
        new_model_checkpoint_path  = saved_model_path.split('/')[:-2] + str(int(time.time())) + '/' 

        new_model_path = new_model_checkpoint_path + model_file_name
        model_checkpoint = ModelCheckpoint(new_model_checkpoint_path + model_file_name,
                                           save_best_only=True, save_weights_only=False)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        tensorboard = TensorBoard(log_dir=new_model_checkpoint_path + "logs/{}".format(time.time()))

        model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
                  epochs=50, batch_size=3, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return new_model_path

def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self._trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

