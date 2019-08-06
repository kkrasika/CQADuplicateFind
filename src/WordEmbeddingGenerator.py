# Import required libraries
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim.models.doc2vec import TaggedDocument
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

def build_model_and_save(questions_labeled, model_name):
    model = Doc2Vec(dm=1, min_count=1, window=10, size=150, sample=1e-4, negative=10)
    model.build_vocab(questions_labeled)
    # Train the model with 20 epochs
    for epoch in range(10):
        model.train(questions_labeled, epochs=model.iter, total_examples=model.corpus_count)
        print(str(model_name)+" Epoch #{} is complete.".format(epoch + 1))

    model.save('../data/model/' + model_name + "-d2v.model")

def main():
    questions_labeled_full = []
    questions_labeled_full_int = []

    fileNameList = ['android', 'english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex',
                    'unix', 'webmasters', 'wordpress']

    for a in range(len(fileNameList)):

        questions_labeled = []
        questions_labeled_int = []

        df = pd.read_json('../data/raw/' + str(fileNameList[a]) + '_questions.json', orient='index')
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

        build_model_and_save(questions_labeled, str(fileNameList[a]))
        build_model_and_save(questions_labeled_int, 'int-'+str(fileNameList[a]))

    build_model_and_save(questions_labeled_full, 'full')
    build_model_and_save(questions_labeled_full_int, 'int-full')

if __name__ == '__main__':
    main()