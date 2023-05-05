import torch

import numpy as np

from dataset import *
from models import *
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def eval():
    csv_file = "../data/trec_test.csv"
    train_df = pd.read_csv(csv_file)
    max_len = np.max(train_df["text"].apply(lambda x: len(x.split())))
    word2idx = json.load(open('../data/word2idx.json', 'r'))

    for i in range(len(train_df)):
        #Tokenize
        train_df['text'].iloc[i] = train_df['text'].iloc[i].lower() 
        train_df['text'].iloc[i] = [word2idx[word] if word in word2idx else word2idx["<unk>"] for word in train_df['text'].iloc[i].split()] #Tokenize
        train_df['text'].iloc[i] = train_df['text'].iloc[i] + [word2idx["<pad>"]] * (max_len - len(train_df['text'].iloc[i]))
        train_df['text'].iloc[i] = torch.tensor(train_df['text'].iloc[i])

    vocab_size = len(word2idx)
    embedding_dim = 128
    output_dim = 2400
    encoder = SentenceEncoder(vocab_size, embedding_dim, output_dim)
    encoder.load_state_dict(torch.load("output/model_encoder.pt"))

    for i in range(len(train_df)):
        #Encode
        train_df['text'].iloc[i] = encoder(train_df['text'].iloc[i].unsqueeze(0)).detach().numpy()
    

    X = np.array(train_df['text'].tolist())
    X = np.squeeze(X, axis=1)
    y = np.array(train_df['coarse_label'].tolist())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    eval()