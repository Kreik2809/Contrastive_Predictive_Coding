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
    dataset = NLPDataset("../data/trec.csv")
    dataset.df = pd.read_csv(dataset.csv_file)
    for i in range(len(dataset.df)):
        dataset.df['text'].iloc[i] = dataset.get_item(i)

    vocab_size = len(json.load(open('../data/word2idx.json', 'r')))
    embedding_dim = 128
    output_dim = 2400
    encoder = SentenceEncoder(vocab_size, embedding_dim, output_dim)
    encoder.load_state_dict(torch.load("model_enc.pt"))

    for i in range(len(dataset.df)):
        dataset.df['text'].iloc[i] = encoder(dataset.df['text'].iloc[i].unsqueeze(0)).detach().numpy()
    
    print(dataset.df.head())


    X = np.array(dataset.df['text'].tolist())
    X = np.squeeze(X, axis=1)
    y = np.array(dataset.df['coarse_label'].tolist())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    eval()