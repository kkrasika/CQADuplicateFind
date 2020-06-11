import pandas as pd
from sklearn import model_selection
from gensim.models import Doc2Vec
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from html.parser import HTMLParser
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from tensorflow.keras import backend as K

def get_df_from_csv_file(fileName):
    print("----- Data frame for domain : " + str(fileName) + ' -----')
    trainDfFromCsv = pd.read_csv('../data/csv/' + fileName + '-training-data.csv', sep=',')
    trainingData = trainDfFromCsv[['Q1','Q1ID', 'Q2','Q2ID', 'Q2Ans', 'Dup', 'DomainId']]
    trainingData = trainingData[:num_of_records_for_domain(fileName)]
    return trainingData

def get_part_df_from_csv_file(fileName, fromid, toid):
    trainDfFromCsv = pd.read_csv('../data/csv/' + fileName + '-training-data.csv', sep=',')
    trainingData = trainDfFromCsv[['Q1','Q1ID','Q1Ans', 'Q2','Q2ID', 'Q2Ans', 'Dup', 'DomainId']]
    trainingData = trainingData[fromid:toid]
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

def get_shuffeled_df_from_csv_files_combined(fileNames, max, batch_size):
    iterations = max/batch_size
    print("----- Data frame for shuffeled combined data : " + str(fileNames)+' -----')
    x=0
    trainingData = None
    while x < iterations:
        from_id = x*batch_size
        to_id = (x+1)*batch_size
        i=0
        while i < len(fileNames):
            if (x < (num_of_records_for_domain(fileNames[i])/batch_size)):
                if trainingData is None:
                    trainingData = get_part_df_from_csv_file(fileNames[i], from_id, to_id)
                else:
                    trainingData = trainingData.append(get_part_df_from_csv_file(fileNames[i], from_id, to_id))
            i += 1
        print("----- Iteration : " + str(x) + ' completed. Size of frame '+str(trainingData.shape[0]))
        x += 1
    return trainingData

# returns train_x, valid_x, train_y, valid_y
def get_train_test_split_of_dataframe(dataFrame, withQid):
    print('Row count :' + str(len(dataFrame.index)))

    dupYDf = dataFrame.loc[dataFrame['Dup'] == 1]
    print('Dup count :' + str(len(dupYDf.index)))

    x = dataFrame[['Q1', 'Q2', 'Q1Ans', 'Q2Ans']]
    y = dataFrame[['Dup', 'DomainId']]

    if withQid:
        x = dataFrame[['Q1','Q1ID','Q1Ans', 'Q2','Q2ID', 'Q2Ans']]

    return model_selection.train_test_split(x, y, test_size= 0.1)
    #return model_selection.train_test_split(x, y)

def review_to_wordlist(review, remove_stopwords=True):
    # Clean the text, with the option to remove stopwords.

    # Convert words to lower case and split them
    words = review.lower().split()

    # Optionally remove stop words (true by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    review_text = " ".join(words)

    # Clean the text
    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", review_text)
    review_text = re.sub(r"\'s", " 's ", review_text)
    review_text = re.sub(r"\'ve", " 've ", review_text)
    review_text = re.sub(r"n\'t", " 't ", review_text)
    review_text = re.sub(r"\'re", " 're ", review_text)
    review_text = re.sub(r"\'d", " 'd ", review_text)
    review_text = re.sub(r"\'ll", " 'll ", review_text)
    review_text = re.sub(r",", " ", review_text)
    review_text = re.sub(r"\.", " ", review_text)
    review_text = re.sub(r"!", " ", review_text)
    review_text = re.sub(r"\(", " ( ", review_text)
    review_text = re.sub(r"\)", " ) ", review_text)
    review_text = re.sub(r"\?", " ", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)

    words = review_text.split()

    # Shorten words to their stems
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in words]

    review_text = " ".join(stemmed_words)

    # Return a list of words
    return (review_text)

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def num_of_records_for_domain(x):
    return {
        'android': 1866,
        'english': 5076,
        'gaming': 3531,
        'gis': 978,
        'mathematica': 1302,
        'physics': 2196,
        'programmers': 2637,
        'stats': 645,
        'tex': 4560,
        'unix': 2466,
        'webmasters': 1899,
        #'wordpress': 864
        'wordpress': 10
    }[x]

'''
    return {
        'android': 3600,
        'english': 4800,
        'gaming': 4800,
        'gis': 2400,
        'mathematica': 3600,
        'physics': 4800,
        'programmers': 3600,
        'stats': 2400,
        'tex': 4800,
        'unix': 4800,
        'webmasters': 2400,
        'wordpress': 1200
    }[x]
    
'''

def num_of_records_for_domain_id(x):
    return {
        1: 1866,
        2: 5076,
        3: 3531,
        4: 978,
        5: 1302,
        6: 2196,
        7: 2637,
        8: 645,
        9: 4560,
        10: 2466,
        11: 1899,
        12: 864
    }[x]

'''
    return {
        1: 3600,
        2: 4800,
        3: 4800,
        4: 2400,
        5: 3600,
        6: 4800,
        7: 3600,
        8: 2400,
        9: 4800,
        10: 4800,
        11: 2400,
        12: 1200
    }[x]
    
'''


def create_tokenizer_from_hub_module(bert_path, sess):
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_text_to_examples(sentences_pair, labels, train_domains_in):
    """Create InputExamples"""
    # Params for bert model and tokenization
    sentences1 = [str(x[0]).lower() for x in sentences_pair]
    sentences2 = [str(x[1]).lower() for x in sentences_pair]

    sentences1_texts = np.array(sentences1, dtype=object)[:, np.newaxis]
    sentences2_texts = np.array(sentences2, dtype=object)[:, np.newaxis]

    InputExamples = []
    for sentences1_text, sentences2_text, label, train_domain in zip(sentences1_texts, sentences2_texts, labels, train_domains_in):
        InputExamples.append(
            InputExample(guid=None, text_a=sentences1_text, text_b=sentences2_text, label=label, domain=train_domain)
        )
    return InputExamples

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, domain=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.domain = domain

def convert_examples_to_features(tokenizer, examples, max_seq_length=256, sent1 = False):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length, sent1
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        list(labels)
    )

def convert_single_example(tokenizer, example, max_seq_length=256, sent1 = True):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    if(sent1):
        text = example.text_a[0]
    else:
        text = example.text_b[0]

    tokens_a = tokenizer.tokenize(text)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """