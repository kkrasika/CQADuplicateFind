# Import required libraries
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from DataSetUtil import review_to_wordlist

def build_model_and_save(questions_labeled, directory, model_name):
    model = Doc2Vec(dm=1, min_count=1, window=10, size=150, sample=1e-4, negative=10)
    model.build_vocab(questions_labeled)
    # Train the model with 20 epochs
    for epoch in range(10):
        model.train(questions_labeled, epochs=model.iter, total_examples=model.corpus_count)
        print(str(model_name)+" Epoch #{} is complete.".format(epoch + 1))

    model.save('../data/model/' + directory + '/' + model_name + "-d2v.model")

def main():
    questions_labeled_full = []
    questions_labeled_full_int = []

    fileNameList = ['android', 'english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex',
                    'unix', 'webmasters', 'wordpress']

    for a in range(len(fileNameList)):

        questions_labeled = []
        questions_labeled_int = []

        df = pd.read_json('../data/json/' + str(fileNameList[a]) + '_questions.json', orient='index')
        df['QuestionID'] = df.index
        df = df[['QuestionID', 'title', 'dups']]

        for i in range(len(df)):

            reviewedTitle = review_to_wordlist(df.iloc[i].loc['title'])

            splittedTitle = reviewedTitle.split()

            docTag = str(df.iloc[i].loc['QuestionID'])
            docTagInt = int(df.iloc[i].loc['QuestionID'])

            questions_labeled.append(TaggedDocument(splittedTitle, [docTag]))
            questions_labeled_int.append(TaggedDocument(splittedTitle, [docTagInt]))
            questions_labeled_full.append(TaggedDocument(splittedTitle, [docTag]))
            questions_labeled_full_int.append(TaggedDocument(splittedTitle, [docTagInt]))

            if i % 1000 == 0:
                progress = i / df['QuestionID'].size * 100
                print("Tagging {}% complete".format(round(progress, 2)) + " of file : " + fileNameList[a])

        build_model_and_save(questions_labeled, 'doc2vec/text-labeled', str(fileNameList[a]), )
        build_model_and_save(questions_labeled_int, 'doc2vec/int-labeled', str(fileNameList[a]))

    build_model_and_save(questions_labeled_full, 'doc2vec/text-labeled', 'full')
    build_model_and_save(questions_labeled_full_int, 'doc2vec/int-labeled', 'full')

if __name__ == '__main__':
    main()