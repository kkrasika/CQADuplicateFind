# This converts json data sets in to csv files which maintains given number of question pairs including duplicates of defined percentage

import pandas as pd
import math

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from html.parser import HTMLParser
import re

recordCount = 1000
duplicatePercentage = 33

def get_records(df, recordids):
    return df.loc[df['QuestionID'].isin(recordids)]

def check_dup(q1, q2):

    q1Dupids = q1.loc['dups']
    q2Dupids = q2.loc['dups']

    return q1.loc['QuestionID'] in q2Dupids and q2.loc['QuestionID'] in q1Dupids

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

def review_to_wordlist(review, remove_stopwords=True):
    resultant_list = []
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

    return review_text

def prepareDataSet(df, non_dup_rows, dup_rows):

    dupCount = 0
    nonDupCount = 0
    maxNonDupCountPerQuestion = 2 #(math.floor(100/duplicatePercentage))-1
    recordCount = len(dup_rows) * 3
    nonDupSize = len(non_dup_rows)

    result = []

    while (dupCount+nonDupCount) < recordCount:
        dupRow = dup_rows[dupCount]
        q1Title = review_to_wordlist(dupRow.loc['title'])
        q1Body = review_to_wordlist(strip_tags(dupRow.loc['body']))
        q1Id = str(dupRow.loc['QuestionID'])

        dupids = dupRow.loc['dups']
        dups = get_records(df, dupids)
        q2Title = review_to_wordlist(dups.iloc[0].loc['title'])
        q2Body = review_to_wordlist(strip_tags(dups.iloc[0].loc['body']))
        q2Id = str(dups.iloc[0].loc['QuestionID'])
        result.append({'Q1ID': q1Id, 'Q1': q1Title+" "+q1Body, 'Q2ID': q2Id, 'Q2': q2Title+" "+q2Body, 'Dup': 1})
        #result.append({'Q1ID': q1Id, 'Q1': q1Title, 'Q2ID': q2Id, 'Q2': q2Title , 'Dup': 1})
        dupCount += 1

        nonDupCountPerQuestion = 0
        nonDupRow = non_dup_rows[nonDupCount]

        if not check_dup(dupRow, nonDupRow):
            q2Title = review_to_wordlist(nonDupRow.loc['title'])
            q2Body = review_to_wordlist(strip_tags(nonDupRow.loc['body']))
            q2Id = str(nonDupRow.loc['QuestionID'])
            result.append({'Q1ID': q1Id, 'Q1': q1Title+" "+q1Body, 'Q2ID': q2Id, 'Q2': q2Title+" "+q2Body, 'Dup': 0})
            # result.append({'Q1ID': q1Id, 'Q1': q1Title, 'Q2ID': q2Id, 'Q2': q2Title, 'Dup': 0})
            nonDupCountPerQuestion += 1
            nonDupCount += 1

        while nonDupCountPerQuestion < maxNonDupCountPerQuestion:
            nonDupRow = non_dup_rows[nonDupCount]
            nonDupRowFromEnd = non_dup_rows[nonDupSize - nonDupCount]
            if not check_dup(nonDupRow, nonDupRowFromEnd):
                q1Title = review_to_wordlist(nonDupRow.loc['title'])
                q1Body = review_to_wordlist(strip_tags(nonDupRow.loc['body']))
                q1Id = str(nonDupRow.loc['QuestionID'])
                q2Title = review_to_wordlist(nonDupRowFromEnd.loc['title'])
                q2Body = review_to_wordlist(strip_tags(nonDupRowFromEnd.loc['body']))
                q2Id  = str(nonDupRowFromEnd.loc['QuestionID'])
                result.append({'Q1ID': q1Id, 'Q1': q1Title+" "+q1Body,'Q2ID': q2Id,  'Q2': q2Title+" "+q2Body, 'Dup': 0})
                #result.append({'Q1ID': q1Id, 'Q1': q1Title, 'Q2ID': q2Id, 'Q2': q2Title, 'Dup': 0})
                nonDupCountPerQuestion += 1
                nonDupCount += 1

        print('Progressing : '+ str(dupCount+nonDupCount)+' / '+ str(recordCount))

    return result

def create_csv_for_file(filename):

    print('Training file generation starts : ' + str(filename))

    df = pd.read_json('../data/json/'+filename+'_questions.json', orient='index')
    df['QuestionID'] = df.index
    df = df[['QuestionID', 'title','body', 'dups']]
    size = df['QuestionID'].size

    dup_rows = []
    non_dup_rows = []

    for i in range(0, size):
        if df.iloc[i].loc['dups']:
            dup_rows.append(df.iloc[i])
        else:
            non_dup_rows.append(df.iloc[i])

    trainingRecords = prepareDataSet(df, non_dup_rows, dup_rows)

    trainDf = pd.DataFrame(columns=['Q1', 'Q2', 'Dup'])

    trainDf = trainDf.append(trainingRecords, ignore_index=True, sort=True)

    trainDf.to_csv('../data/csv/'+str(filename)+'-training-data.csv', sep=',')

    print('Training file generation completed : ' + str(filename))

def create_csv_for_files(fileNames):
    for fileName in fileNames:
        create_csv_for_file(fileName)

def main():
    fileNameList = ['android','english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
    #fileNameList = ['webmasters']
    create_csv_for_files(fileNameList)

if __name__ == '__main__':
    main()
