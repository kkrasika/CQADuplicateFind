from rank_bm25 import BM25Okapi
import pandas as pd
import gc
from DataSetUtil import get_df_from_csv_file, review_to_wordlist
from SiameseLSTMClassifier import get_doc2vec_vectors_train_valid_split, train_model, evaluate_model
from keras.models import load_model
from inputHandler import create_test_data
from config import siamese_config
from operator import itemgetter
from DataSetUtil import strip_tags
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer)
import torch
import numpy as np

def main():
    numberOfRelevantQs = 10
    fileNameList = ['android','english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
    #fileNameList = ['stats', 'tex', 'unix', 'webmasters', 'wordpress']
    fileNameList = ['webmasters']

    for a in range(len(fileNameList)):

        outputFile = open('../data/output/result4.txt', 'a')

        fileName = str(fileNameList[a])

        # df_for_file = get_df_from_csv_file(fileName)
        # train_x, train_y, valid_x, valid_y, embedding_meta_data = get_doc2vec_vectors_train_valid_split(df_for_file)
        # model_path = train_model(train_x, train_y, embedding_meta_data, fileName)
        # siamese_lstm_model = load_model(model_path)
        # preds, accuracy = evaluate_model(siamese_lstm_model, valid_x, valid_y)
        # print('Classification Accuracy for : ' + fileName + ' Siamese LSTM ' + str(str(accuracy[1])), file=outputFile)
        # tokenizer = embedding_meta_data['tokenizer']

        xlnet_out_address = '../data/model/xlnet/'+fileName
        tag2idx = {'0': 0, '1': 1}
        model = XLNetForSequenceClassification.from_pretrained(xlnet_out_address, num_labels=len(tag2idx))

        corpus = []
        corpus_to_query = []
        df = pd.read_json('../data/json/' + str(fileName) + '_questions.json', orient='index')
        df['QuestionID'] = df.index
        df = df[['QuestionID', 'title', 'body', 'dups']]
        #df = df[:100]
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

                candidate_query = review_to_wordlist(df.iloc[i].loc['title'])

                queriesWithDuplicates += 1

                candidate_docs_bm25 = bm25.get_top_n(tokenized_query, corpus_to_query, n=20)
                topn_similar_docs_as_pairs = []
                topn_similar_docs_as_pairs2 = []
                topn_doc_indexes = []


                for p in range(0, len(candidate_docs_bm25)):
                    candidateQuestion = review_to_wordlist(candidate_docs_bm25[p][1])
                    topn_similar_docs_as_pairs.append(candidate_query+ ' ' +candidateQuestion)
                    topn_similar_docs_as_pairs2.append((candidate_query,candidateQuestion))
                    topn_doc_indexes.append(candidate_docs_bm25[p][0])

                b_input_ids, b_segs, b_input_mask, b_labels = getValidationData(topn_similar_docs_as_pairs)
                with torch.no_grad():
                    outputs = model(input_ids=b_input_ids, token_type_ids=b_segs, input_mask=b_input_mask)
                    logits = outputs[:2]

                # Get textclassification predict result
                logits = logits[0].detach().cpu().numpy()
                #label_ids = b_labels.to('cpu').numpy()

                results = [(x, y, z) for (x, y), z in zip(topn_similar_docs_as_pairs2, logits[...,0]+logits[...,1])]
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
                    print(str(foundDupsAll) + ' - ' + str(queriesWithDuplicates))
                precisionAtSum += foundDups / numberOfRelevantQs
                foundDupsAll += foundDups

        print(str(fileName) + " Train Data Count : " + str(numberOfTrainData), file=outputFile)
        print(str(fileName) + " Relevant Query Count (Test) : " + str(queriesWithDuplicates), file=outputFile)
        print(str(fileName) + " Found within " + str(numberOfRelevantQs) + " Count : " + str(foundDupsAll), file=outputFile)
        print(str(fileName) + " Precision @ " + str(numberOfRelevantQs) + " : " + str(precisionAtSum / queriesWithDuplicates), file=outputFile)
        print(str(fileName) + " MAP : " + str(sumOfAveragePrecision / queriesWithDuplicates), file=outputFile)

        outputFile.close()
        del df
        del bm25
        del corpus
        del corpus_to_query
        del topn_similar_docs_as_pairs
        gc.collect()

def getValidationData(sentences):
    vocabulary = '../data/model/xlnet/xlnet-base-cased-spiece.model'
    tag2idx = {'0': 0, '1': 1}
    max_len = 32

    tokenizer = XLNetTokenizer(vocab_file=vocabulary, do_lower_case=False)

    full_input_ids = []
    full_input_masks = []
    full_segment_ids = []
    full_val_tags = []

    SEG_ID_A = 0
    SEG_ID_B = 1
    SEG_ID_CLS = 2
    SEG_ID_SEP = 3
    SEG_ID_PAD = 4

    UNK_ID = tokenizer.encode("<unk>")[0]
    CLS_ID = tokenizer.encode("<cls>")[0]
    SEP_ID = tokenizer.encode("<sep>")[0]
    MASK_ID = tokenizer.encode("<mask>")[0]
    EOD_ID = tokenizer.encode("<eod>")[0]

    for i, sentence in enumerate(sentences):
    # Tokenize sentence to token id list
        tokens_a = tokenizer.encode(sentence)

        # Trim the len of text
        if (len(tokens_a) > max_len - 2):
            tokens_a = tokens_a[:max_len - 2]

        tokens = []
        segment_ids = []

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(SEG_ID_A)

        # Add <sep> token
        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_A)

        # Add <cls> token
        tokens.append(CLS_ID)
        segment_ids.append(SEG_ID_CLS)

        input_ids = tokens

        # The mask has 0 for real tokens and 1 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [0] * len(input_ids)

        # Zero-pad up to the sequence length at fornt
        if len(input_ids) < max_len:
            delta_len = max_len - len(input_ids)
            input_ids = [0] * delta_len + input_ids
            input_mask = [1] * delta_len + input_mask
            segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        assert len(segment_ids) == max_len

        full_input_ids.append(input_ids)
        full_input_masks.append(input_mask)
        full_segment_ids.append(segment_ids)
        full_val_tags.append(0)

        # if 3 > i:
        #     print("No.:%d" % (i))
        #     print("sentence: %s" % (sentence))
        #     print("input_ids:%s" % (input_ids))
        #     print("attention_masks:%s" % (input_mask))
        #     print("segment_ids:%s" % (segment_ids))
        #     print("\n")

    val_inputs = torch.tensor(full_input_ids)
    val_masks = torch.tensor(full_input_masks)
    val_segs = torch.tensor(full_segment_ids)
    val_tags = torch.tensor(full_val_tags)

    batch_num = 16

    valid_data = TensorDataset(val_inputs, val_masks, val_segs, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_num)

    return val_inputs, val_masks, val_segs, val_tags

if __name__ == '__main__':
    main()