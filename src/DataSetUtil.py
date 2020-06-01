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
        'wordpress': 864
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