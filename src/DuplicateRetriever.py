# Import required libraries
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim.models import Doc2Vec

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

## Test for correctness
numberOfRelevantQs = 5

#fileNameList = ['android', 'english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress', 'full']

fileNameList = [ 'webmasters', 'wordpress', 'full']

for a in range(len(fileNameList)):
    correctCount = 0
    dupCount = 0
    try:
        model = Doc2Vec.load('../data/model/'+str(fileNameList[a])+"-d2v.model")
    except FileNotFoundError as e:
        print(e.errno)

    df = pd.read_json('../data/raw/' + str(fileNameList[a]) + '_questions.json', orient='index')
    df['QuestionID'] = df.index
    df = df[['QuestionID', 'title', 'dups']]
    numberOfTestData = df['QuestionID'].size

    for i in range(0,numberOfTestData):

        dupids = df.iloc[i].loc['dups']

        test_data = review_to_wordlist(df.iloc[i].loc['title'])
        #test_data = df.iloc[i].loc['title']
        v1 = model.infer_vector(test_data.split())

        if(len(dupids)>0):
            dupCount += 1
            qid = df.iloc[i].loc['QuestionID']

            vec = model.docvecs[str(qid)]
            #print("Dups of  " + str(qid) + " : ", dupids)
            # to find most similar doc using tags
            similar_doc = model.docvecs.most_similar([v1], topn=numberOfRelevantQs)
            #print("Similar doc :" + str(similar_doc))

            for j in range(0, numberOfRelevantQs):
                similarDoc = similar_doc[j]
                if(similarDoc[0] in dupids):
                    correctCount += 1

    print(str(fileNameList[a])+" Test Data Count : " + str(numberOfTestData))
    print(str(fileNameList[a])+" Duplicate Count : " + str(dupCount))
    print(str(fileNameList[a])+" Correct Count : " + str(correctCount))
    print(str(fileNameList[a]) + " Accuracy : " + str(correctCount*100/dupCount))

#print(model.docvecs['1'])