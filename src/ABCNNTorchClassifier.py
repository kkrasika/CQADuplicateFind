from ABCNN import Abcnn3
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import DataLoader
from DataSetUtil import get_df_from_csv_file, review_to_wordlist
from SiameseLSTMClassifier import get_doc2vec_vectors_train_valid_split, train_model, evaluate_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np


model = Abcnn3(emb_dim=150, sentence_length=80, filter_width=50)

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
num_epochs = 10
filename = 'webmasters'

class DupStackDataset(Dataset):
    def __init__(self):
        self.x1 = []
        self.x2 = []
        self.y = []

        left_seq_len = 80
        right_seq_len = 80

        embed_dimensions = 150

        df_for_file = get_df_from_csv_file(filename)
        train_x, train_y, valid_x, valid_y, embedding_meta_data = get_doc2vec_vectors_train_valid_split(df_for_file)

        tokenizer, embedding_matrix = embedding_meta_data['tokenizer'], embedding_meta_data['embedding_matrix']

        # Training data
        sentences1 = [str(x[0]).lower() for x in train_x]
        sentences2 = [str(x[1]).lower() for x in train_x]
        train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
        train_sequences_2 = tokenizer.texts_to_sequences(sentences2)

        train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=left_seq_len)
        train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=right_seq_len)

        embedx1 = []
        for x in range(0, train_padded_data_1.shape[0]):
            embeds = []
            for y in range(0, train_padded_data_1.shape[1]):
                embed = embedding_matrix[train_padded_data_1[x, y]]
                embeds.append(embed)
            embedx1.append(embeds)

        embedx2 = []
        for x in range(0, train_padded_data_2.shape[0]):
            embeds = []
            for y in range(0, train_padded_data_2.shape[1]):
                embed = embedding_matrix[train_padded_data_2[x, y]]
                embeds.append(embed)
            embedx2.append(embeds)

        self.x1 = embedx1
        self.x2 = embedx2

        self.y = np.array(train_y)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        x1 = self.x1[idx]
        x2 = self.x2[idx]
        y = self.y[idx]

        x1tensor = torch.Tensor(x1)  # torch.tensor(np.stack(x1)) # torch.from_numpy(x1)
        x2tensor = torch.Tensor(x2)  # torch.tensor(np.stack(x2)) # torch.from_numpy(x2)

        x1tensor = x1tensor.unsqueeze(0)  # torch.tensor(np.stack(x1)) # torch.from_numpy(x1)
        x2tensor = x2tensor.unsqueeze(0)  # torch.tensor(np.stack(x2)) # torch.from_numpy(x2)

        return (x1tensor, x2tensor, y)


if __name__ == '__main__':
    dataset = DupStackDataset()
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=4)

    # Train the model
    total_step = len(dataloader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (x1, x2, labels) in enumerate(dataloader):

            outputs = model.predict(x2, x1)
            labels = labels.unsqueeze(1)

            print('Outputs : ' + str(outputs))

            loss = criterion(outputs.float(), labels.float())
            loss_list.append(loss.item())

            print('Labels : ' + str(labels))

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))