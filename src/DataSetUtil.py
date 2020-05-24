import pandas as pd
from sklearn import model_selection
from gensim.models import Doc2Vec
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from html.parser import HTMLParser

def get_df_from_csv_file(fileName):
    print("----- Data frame for domain : " + str(fileName) + ' -----')
    trainDfFromCsv = pd.read_csv('../data/csv/' + fileName + '-training-data.csv', sep=',')
    trainingData = trainDfFromCsv[['Q1','Q1ID', 'Q2','Q2ID', 'Dup', 'DomainId']]
    trainingData = trainingData[:4800]
    return trainingData

def get_part_df_from_csv_file(fileName, fromid, toid):
    trainDfFromCsv = pd.read_csv('../data/csv/' + fileName + '-training-data.csv', sep=',')
    trainingData = trainDfFromCsv[['Q1','Q1ID', 'Q2','Q2ID', 'Q2Ans', 'Dup', 'DomainId']]
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

    x = dataFrame[['Q1', 'Q2', 'Q2Ans']]
    y = dataFrame[['Dup', 'DomainId']]

    if withQid:
        x = dataFrame[['Q1','Q1ID', 'Q2','Q2ID', 'Q2Ans']]

    return model_selection.train_test_split(x, y, test_size= 0.1)

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
        'gaming': 978,
        'gis': 1302,
        'mathematica': 2196,
        'physics': 2637,
        'programmers': 645,
        'stats': 4560,
        'tex': 2466,
        'unix': 2466,
        'webmasters': 1899,
        'wordpress': 864
    }[x]

    '''
    return {
        'android' : 3600,
        'english' : 2400,
        'gaming' : 2400,
        'gis' : 2400,
        'mathematica' : 2400,
        'physics' : 4800,
        'programmers' : 2400,
        'stats' : 2400,
        'tex' : 2400,
        'unix' : 2400,
        'webmasters' : 2400,
        'wordpress' : 1200
    }[x]
    '''


