import numpy as np
from keras_abcnn import ABCNN
from DataSetUtil import get_df_from_csv_file, review_to_wordlist
from SiameseLSTMClassifier import get_doc2vec_vectors_train_valid_split, train_model, evaluate_model
from keras.preprocessing.sequence import pad_sequences

fileNameList = ['android','english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
#fileNameList = ['webmasters']

for fileName in fileNameList:

    outputFile = open('../data/output/result2.txt', 'a')

    left_seq_len = 80
    right_seq_len = 80

    embed_dimensions = 150

    nb_filter = 150
    filter_width = 10


    df_for_file = get_df_from_csv_file(fileName)
    train_x, train_y, valid_x, valid_y, embedding_meta_data = get_doc2vec_vectors_train_valid_split(df_for_file)

    tokenizer, embedding_matrix = embedding_meta_data['tokenizer'], embedding_meta_data['embedding_matrix']

    # Training data
    sentences1 = [str(x[0]).lower() for x in train_x]
    sentences2 = [str(x[1]).lower() for x in train_x]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=left_seq_len)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=right_seq_len)

    train_labels = np.array(train_y)
    num_samples = len(train_labels)
    Y = np.random.randint(0, 2, (num_samples,))
    Y = train_labels

    x1 = np.random.random(size=(num_samples, left_seq_len, embed_dimensions))
    x2 = np.random.random(size=(num_samples, right_seq_len, embed_dimensions))

    for x in range(0, train_padded_data_1.shape[0]):
        for y in range(0, train_padded_data_1.shape[1]):
            embed = embedding_matrix[train_padded_data_1[x, y]]
            for z in range(0, len(embed)):
                x1[x,y,z]=embed[z]

    for x in range(0, train_padded_data_2.shape[0]):
        for y in range(0, train_padded_data_2.shape[1]):
            embed = embedding_matrix[train_padded_data_2[x, y]]
            for z in range(0, len(embed)):
                x1[x, y, z] = embed[z]

    X = [x1,x2]

    # _plot_all(left_seq_len, right_seq_len, embed_dimensions, nb_filter, filter_width)

    model = ABCNN(
        left_seq_len=left_seq_len, right_seq_len=right_seq_len, depth=5,
        embed_dimensions=embed_dimensions, nb_filter=nb_filter, filter_widths=filter_width,
        collect_sentence_representations=True, abcnn_1=True, abcnn_2=True,
        # mode="euclidean",
        mode="dot"
    )

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    model.fit(X, Y, nb_epoch=30)

    # Test data
    sentences1 = [str(x[0]).lower() for x in valid_x]
    sentences2 = [str(x[1]).lower() for x in valid_x]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=left_seq_len)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=right_seq_len)

    valid_labels = np.array(valid_y)
    num_samples = len(valid_labels)
    Y = np.random.randint(0, 2, (num_samples,))
    Y = valid_labels

    xv1 = np.random.random(size=(num_samples, left_seq_len, embed_dimensions))
    xv2 = np.random.random(size=(num_samples, right_seq_len, embed_dimensions))

    for x in range(0, train_padded_data_1.shape[0]):
        for y in range(0, train_padded_data_1.shape[1]):
            embed = embedding_matrix[train_padded_data_1[x, y]]
            for z in range(0, len(embed)):
                xv1[x, y, z] = embed[z]

    for x in range(0, train_padded_data_2.shape[0]):
        for y in range(0, train_padded_data_2.shape[1]):
            embed = embedding_matrix[train_padded_data_2[x, y]]
            for z in range(0, len(embed)):
                xv1[x, y, z] = embed[z]

    XV = [xv1, xv2]
    score = model.evaluate(XV, Y, verbose=0)
    preds = list(model.predict(XV, verbose=0).ravel())

    print('Predicts : '+str(preds))

    print('Classification Accuracy for : ' + fileName + ' BCNN ' + str(str(score[1])), file=outputFile)
    outputFile.close()