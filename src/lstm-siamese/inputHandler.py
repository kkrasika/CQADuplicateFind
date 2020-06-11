from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from gensim.models import FastText
import numpy as np
import gc
import pandas as pd
from DataSetUtil import create_tokenizer_from_hub_module, convert_text_to_examples, convert_examples_to_features
import tensorflow as tf
from transformers import (TFXLNetModel, XLNetTokenizer)
from copy import copy
from tqdm import tqdm

def train_word2vec(documents, embedding_dim):
    """
    train word2vector over traning documents
    Args:
        documents (list): list of document
        embedding_dim (int): outpu wordvector sizeN
    Returns:
        word_vectors(dict): dict containing words and their respective vectors
    """
    model = Word2Vec(documents, min_count=1, size=embedding_dim)
    word_vectors = model.wv
    del model
    return word_vectors


def create_embedding_matrix(tokenizer, word_vectors, embedding_dim):
    """
    Create embedding matrix containing word indexes and respective vectors from word vectors
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object containing word indexes
        word_vectors (dict): dict containing word and their respective vectors
        embedding_dim (int): dimention of word vector

    Returns:

    """
    nb_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))
    for word, i in word_index.items():
        try:
            embedding_vector = word_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except KeyError:
            print("vector not found for word - %s" % word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix


def word_embed_meta_data(documents, embedding_dim):
    """
    Load tokenizer object for given vocabs list
    Args:
        documents (list): list of document
        embedding_dim (int): embedding dimension
    Returns:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        embedding_matrix (dict): dict with word_index and vector mapping
    """
    documents = [str(x).lower().split() for x in documents]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(documents)
    word_vector = train_word2vec(documents, embedding_dim)
    embedding_matrix = create_embedding_matrix(tokenizer, word_vector, embedding_dim)
    del word_vector
    gc.collect()
    return tokenizer, embedding_matrix


def create_train_dev_set(tokenizer, sentences_pair, is_similar, train_domains_in, max_sequence_length, validation_split_ratio):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        sentences_pair (list): list of tuple of sentences pairs
        is_similar (list): list containing labels if respective sentences in sentence1 and sentence2
                           are same or not (1 if same else 0)
        max_sequence_length (int): max sequence length of sentences to apply padding
        validation_split_ratio (float): contain ratio to split training data into validation data

    Returns:
        train_data_1 (list): list of input features for training set from sentences1
        train_data_2 (list): list of input features for training set from sentences2
        labels_train (np.array): array containing similarity score for training data
        leaks_train(np.array): array of training leaks features

        val_data_1 (list): list of input features for validation set from sentences1
        val_data_2 (list): list of input features for validation set from sentences1
        labels_val (np.array): array containing similarity score for validation data
        leaks_val (np.array): array of validation leaks features
    """
    sentences1 = [str(x[0]).lower() for x in sentences_pair]
    sentences2 = [str(x[1]).lower() for x in sentences_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(train_sequences_1, train_sequences_2)]

    df = pd.DataFrame.from_records(leaks)
    df.columns = ['a', 'b','c']
    df.groupby(['a']).agg(['count'])
    df.groupby(['b']).agg(['count'])

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)
    train_labels = np.array(is_similar)
    train_domains = np.array(train_domains_in)
    leaks = np.array(leaks)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    train_domains_shuffled = train_domains[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1
    del train_padded_data_2
    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    domains_train, domains_val = train_domains_shuffled[:-dev_idx], train_domains_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, domains_train, leaks_train, val_data_1, val_data_2, labels_val, domains_val, leaks_val


def create_test_data(tokenizer, test_sentences_pair, max_sequence_length):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        test_sentences_pair (list): list of tuple of sentences pairs
        max_sequence_length (int): max sequence length of sentences to apply padding

    Returns:
        test_data_1 (list): list of input features for training set from sentences1
        test_data_2 (list): list of input features for training set from sentences2
    """
    test_sentences1 = [str(x[0]).lower() for x in test_sentences_pair]
    test_sentences2 = [str(x[1]).lower() for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                  for x1, x2 in zip(test_sequences_1, test_sequences_2)]

    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)

    return test_data_1, test_data_2, leaks_test

def create_train_dev_set_for_bert(sentences_pair, is_similar, train_domains_in, max_sequence_length, sess):

    bert_path = "../data/model/bert_uncased_L-12_H-768_A-12_1"

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module(bert_path, sess)

    # Convert data to InputExample format
    train_examples = convert_text_to_examples(sentences_pair, is_similar, train_domains_in)

    # Convert to features
    (
        train_input_ids,
        train_input_masks,
        train_segment_ids,
        train_labels
    ) = convert_examples_to_features(
        tokenizer, train_examples, max_seq_length=max_sequence_length, sent1 =True
    )

    # Convert to features
    (
        train_input2_ids,
        train_input2_masks,
        train_segment2_ids,
        train_labels2
    ) = convert_examples_to_features(
        tokenizer, train_examples, max_seq_length=max_sequence_length
    )

    return train_input_ids, train_input_masks, train_segment_ids, train_input2_ids, train_input2_masks, train_segment2_ids, train_labels, train_domains_in

def create_test_data_for_bert(test_sentences_pair, is_similar_validate, valid_domain_list, max_sequence_length, sess):
    bert_path = "../data/model/bert_uncased_L-12_H-768_A-12_1"

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module(bert_path, sess)

    # Convert data to InputExample format
    train_examples = convert_text_to_examples(test_sentences_pair, is_similar_validate, valid_domain_list)

    # Convert to features
    (
        valid_input_ids,
        valid_input_masks,
        valid_segment_ids,
        valid_labels
    ) = convert_examples_to_features(
        tokenizer, train_examples, max_seq_length=max_sequence_length, sent1 =True
    )

    # Convert to features
    (
        valid_input2_ids,
        valid_input2_masks,
        valid_segment2_ids,
        valid_labels2
    ) = convert_examples_to_features(
        tokenizer, train_examples, max_seq_length=max_sequence_length
    )

    return valid_input_ids, valid_input_masks, valid_segment_ids, valid_input2_ids, valid_input2_masks, valid_segment2_ids, valid_labels, valid_domain_list


def create_train_dev_set_for_xlnet(sentences_pair, is_similar, train_domains_in, max_sequence_length):

    model_file_address = '../data/model/xlnet/xlnet-base-cased'
    tokenizer = XLNetTokenizer.from_pretrained(model_file_address)

    input1_ids, input2_ids = [], []
    for sentence in tqdm(sentences_pair, desc="Converting examples to features"):

        encode1 = tokenizer.encode(str(sentence[0]).lower(), add_special_tokens=True)
        input1_ids.append(encode1)
        encode2 = tokenizer.encode(str(sentence[1]).lower(), add_special_tokens=True)
        input2_ids.append(encode2)

    return (
        pad_sequences(input1_ids, maxlen=max_sequence_length),
        pad_sequences(input2_ids, maxlen=max_sequence_length),
        np.array(list(is_similar)),
        np.array(train_domains_in)
    )

def create_test_data_for_xlnet(test_sentences_pair, is_similar_validate, valid_domain_list, max_sequence_length):

    model_file_address = '../data/model/xlnet/xlnet-base-cased'
    tokenizer = XLNetTokenizer.from_pretrained(model_file_address)

    input1_ids, input2_ids = [], []
    for sentence in tqdm(test_sentences_pair, desc="Converting examples to features"):

        encode1 = tokenizer.encode(str(sentence[0]).lower(), add_special_tokens=True)
        input1_ids.append(encode1)
        encode2 = tokenizer.encode(str(sentence[1]).lower(), add_special_tokens=True)
        input2_ids.append(encode2)

    return (
        pad_sequences(input1_ids, maxlen=max_sequence_length),
        pad_sequences(input2_ids, maxlen=max_sequence_length),
        np.array(list(is_similar_validate)),
        np.array(valid_domain_list)
    )
