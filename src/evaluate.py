import torch

import numpy as np

from dataset import *
from models import *
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def eval(encoder, train_df, test_df):
    #make copy of df
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()
    for i in range(len(train_df_copy)):
        #Encode
        train_df_copy['text'].iloc[i] = encoder(train_df_copy['text'].iloc[i].unsqueeze(0)).detach().numpy()
    
    X_train = np.array(train_df_copy['text'].tolist())
    X_train = np.squeeze(X_train, axis=1)
    y_train = np.array(train_df_copy['coarse_label'].tolist())

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    clf.fit(X_train, y_train)


    for i in range(len(test_df)):
        #Encode
        test_df_copy['text'].iloc[i] = encoder(test_df_copy['text'].iloc[i].unsqueeze(0)).detach().numpy()
    
    X_test = np.array(test_df_copy['text'].tolist())
    X_test = np.squeeze(X_test, axis=1)
    y_test = np.array(test_df_copy['coarse_label'].tolist())

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return train_acc, test_acc

def main():
    csv_file_train = "../data/trec_train.csv"
    csv_file_test = "../data/trec_test.csv"
    train_df = pd.read_csv(csv_file_train)
    test_df = pd.read_csv(csv_file_test)
    max_len_train = np.max(train_df["text"].apply(lambda x: len(x.split())))
    max_len_test = np.max(test_df["text"].apply(lambda x: len(x.split())))
    word2idx = json.load(open('../data/word2idx.json', 'r'))

    for i in range(len(train_df)):
        #Tokenize
        train_df['text'].iloc[i] = train_df['text'].iloc[i].lower() 
        train_df['text'].iloc[i] = [word2idx[word] if word in word2idx else word2idx["<unk>"] for word in train_df['text'].iloc[i].split()] #Tokenize
        train_df['text'].iloc[i] = train_df['text'].iloc[i] + [word2idx["<pad>"]] * (max_len_train - len(train_df['text'].iloc[i]))
        train_df['text'].iloc[i] = torch.tensor(train_df['text'].iloc[i])

    for i in range(len(test_df)):
        #Tokenize
        test_df['text'].iloc[i] = test_df['text'].iloc[i].lower() 
        test_df['text'].iloc[i] = [word2idx[word] if word in word2idx else word2idx["<unk>"] for word in test_df['text'].iloc[i].split()]
        test_df['text'].iloc[i] = test_df['text'].iloc[i] + [word2idx["<pad>"]] * (max_len_test - len(test_df['text'].iloc[i]))
        test_df['text'].iloc[i] = torch.tensor(test_df['text'].iloc[i])


    vocab_size = len(word2idx)
    embedding_dim = 128
    output_dim = 2400
    encoder = SentenceEncoder(vocab_size, embedding_dim, output_dim)
    def init_weights(m):
        for name, param in m.named_parameters():
            torch.nn.init.uniform_(param.data, -0.08, 0.08)
    encoder.apply(init_weights)

    random_train_acc, random_test_acc = eval(encoder, train_df, test_df)

    #load encoder from '../models/encoder.pt'
    encoder.load_state_dict(torch.load('output/model_encoder.pt', map_location=torch.device('cpu')))
    train_acc, test_acc = eval(encoder, train_df, test_df)

    #plot a table in output folder with the results
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=[[random_train_acc, random_test_acc], [train_acc, test_acc]], rowLabels=['Random', 'Trained'], colLabels=['Train accuracy', 'Test accuracy'], loc='center')
    fig.tight_layout()
    plt.savefig('output/table.png')


if __name__ == "__main__":
    main()