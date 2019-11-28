# Import required libraries
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim.models import Doc2Vec
from DataSetUtil import review_to_wordlist

## Test for correctness
numberOfRelevantQs = 10

#fileNameList = ['android', 'english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress', 'full']

fileNameList = [ 'english' ]

for a in range(len(fileNameList)):
    foundDupsAll = 0
    queriesWithDuplicates = 0
    sumOfAveragePrecision = 0.0
    precisionAtSum = 0

    try:
        model = Doc2Vec.load('../data/model/doc2vec/text-labeled/'+str(fileNameList[a])+"-d2v.model")
    except FileNotFoundError as e:
        print(e.errno)

    df = pd.read_json('../data/json/' + str(fileNameList[a]) + '_questions.json', orient='index')
    df['QuestionID'] = df.index
    df = df[['QuestionID', 'title', 'dups']]
    numberOfTestData = df['QuestionID'].size

    for i in range(0,numberOfTestData):

        dupids = df.iloc[i].loc['dups']

        test_data = review_to_wordlist(df.iloc[i].loc['title'])
        #test_data = df.iloc[i].loc['title']
        v1 = model.infer_vector(test_data.split())

        if(len(dupids)>0):
            queriesWithDuplicates += 1
            qid = df.iloc[i].loc['QuestionID']

            vec = model.docvecs[str(qid)]
            #print("Dups of  " + str(qid) + " : ", dupids)
            # to find most similar doc using tags
            similar_doc = model.docvecs.most_similar([v1], topn=numberOfRelevantQs)
            #print("Similar doc :" + str(similar_doc))
            foundDups = 0
            apForQuery = 0.0
            for j in range(0, numberOfRelevantQs):
                similarDoc = similar_doc[j]
                if(similarDoc[0] in dupids):
                    foundDups += 1
                    apForQuery += foundDups/(j+1)

            if(foundDups>0):
                sumOfAveragePrecision = sumOfAveragePrecision + (apForQuery/foundDups)
            precisionAtSum += foundDups / numberOfRelevantQs
            foundDupsAll+=foundDups

    print(str(fileNameList[a])+" Test Data Count : " + str(numberOfTestData))
    print(str(fileNameList[a])+" Relevant Query Count : " + str(queriesWithDuplicates))
    print(str(fileNameList[a]) + " Found within "+str(numberOfRelevantQs)+" Count : " + str(foundDupsAll))
    print(str(fileNameList[a]) + " Precision @ "+str(numberOfRelevantQs)+" : " + str(precisionAtSum/queriesWithDuplicates))
    print(str(fileNameList[a]) + " Accuracy : " + str(sumOfAveragePrecision/queriesWithDuplicates))

#print(model.docvecs['1'])