from rank_bm25 import BM25Okapi
import pandas as pd
import gc
from DataSetUtil import get_df_from_csv_file, review_to_wordlist
from SiameseLSTMClassifier import get_doc2vec_vectors_train_valid_split, train_model, evaluate_model
from tensorflow.keras.models import load_model
from inputHandler import create_test_data
from config import siamese_config
from operator import itemgetter
from DataSetUtil import strip_tags
from layers.attention import AttentionLayer
from model import GradientReversal

def main():
    numberOfRelevantQs = 10
    #fileNameList = ['android','english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
    fileNameList = ['android','english', 'gaming', 'gis', 'wordpress']
    #fileNameList = ['webmasters']

    for a in range(len(fileNameList)):

        outputFile = open('../data/output/result4.txt', 'a')

        fileName = str(fileNameList[a])

        df_for_file = get_df_from_csv_file(fileName)
        train_x, train_y, train_domain_list, valid_x, valid_y, valid_domain_list, embedding_meta_data = get_doc2vec_vectors_train_valid_split(df_for_file)
        #model_path = train_model(train_x, train_y, embedding_meta_data, fileName)
        #model_path = '../data/model/siamese-lstm/' + 'common-adapted' + '-' + siamese_config['MODEL_FILE_NAME']
        model_path = '../data/model/siamese-lstm/best-to-date/' + 'common-adapted-SiameseLSTM.h5'
        siamese_lstm_model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer, 'GradientReversal': GradientReversal})
        tokenizer = embedding_meta_data['tokenizer']

        corpus = []
        corpus_to_query = []
        df = pd.read_json('../data/json/' + str(fileName) + '_questions.json', orient='index')
        df['QuestionID'] = df.index
        df = df[['QuestionID', 'title', 'body', 'dups']]
        numberOfTrainData = df['QuestionID'].size

        for i in range(0,numberOfTrainData):
            title = review_to_wordlist(df.iloc[i].loc['title'])
            body = review_to_wordlist(strip_tags(df.iloc[i].loc['body']))
            qid = df.iloc[i].loc['QuestionID']
            corpus.append(title)
            corpus_to_query.append((str(qid),title, body))

        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        foundDupsAll = 0
        queriesWithDuplicates = 0
        sumOfAveragePrecision = 0.0
        precisionAtSum = 0
        for i in range(0,numberOfTrainData):

            dupids = df.iloc[i].loc['dups']
            if(len(dupids) > 0):

                test_query = review_to_wordlist(df.iloc[i].loc['title']) +" "+ review_to_wordlist(strip_tags(df.iloc[i].loc['body']))
                tokenized_query = test_query.split(" ")

                queriesWithDuplicates += 1

                candidate_docs_bm25 = bm25.get_top_n(tokenized_query, corpus_to_query, n=20)
                topn_similar_docs_as_pairs = []
                topn_doc_indexes = []


                for p in range(0, len(candidate_docs_bm25)):
                    candidateQuestion = review_to_wordlist(candidate_docs_bm25[p][1]) + " " + review_to_wordlist(strip_tags(candidate_docs_bm25[p][2]))
                    topn_similar_docs_as_pairs.append((test_query,candidateQuestion))
                    topn_doc_indexes.append(candidate_docs_bm25[p][0])

                test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, topn_similar_docs_as_pairs, siamese_config['MAX_SEQUENCE_LENGTH'])
                preds = list(siamese_lstm_model.predict([test_data_x1, test_data_x2, leaks_test], verbose=0)[0].ravel())
                results = [(x, y, z) for (x, y), z in zip(topn_similar_docs_as_pairs, preds)]
                results = [(a, y, z) for (x, y, z), a in zip(results, topn_doc_indexes)]
                results.sort(key=itemgetter(2), reverse=True) 


                foundDups = 0
                apForQuery = 0.0
                #print("Query : " + test_query)
                #print("Annotated Dups : " + str(dupids))
                #print('Results : ')
                for j in range(0, numberOfRelevantQs):
                    similarDocId = results[j][0]
                    #similarDocId = candidate_docs_bm25[j][0]
                    #print(similarDocId)
                    if(similarDocId in dupids):
                        foundDups += 1
                        apForQuery += foundDups/(j+1)
                        #print('-------------- Hit !!!-----------------@'+ str(j))

                if(foundDups>0):
                    sumOfAveragePrecision = sumOfAveragePrecision + (apForQuery/len(dupids))
                precisionAtSum += foundDups / numberOfRelevantQs
                foundDupsAll += foundDups

        print(str(fileName) + " Train Data Count : " + str(numberOfTrainData), file=outputFile)
        print(str(fileName) + " Relevant Query Count (Test) : " + str(queriesWithDuplicates), file=outputFile)
        print(str(fileName) + " Found within " + str(numberOfRelevantQs) + " Count : " + str(foundDupsAll), file=outputFile)
        print(str(fileName) + " Precision @ " + str(numberOfRelevantQs) + " : " + str(precisionAtSum / queriesWithDuplicates), file=outputFile)
        print(str(fileName) + " MAP : " + str(sumOfAveragePrecision / queriesWithDuplicates), file=outputFile)

        outputFile.close()
        del df
        del siamese_lstm_model
        del bm25
        del corpus
        del corpus_to_query
        del topn_similar_docs_as_pairs
        gc.collect()

if __name__ == '__main__':
    main()