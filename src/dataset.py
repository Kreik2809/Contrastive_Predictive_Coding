import os
import random
import json
import torch

import pandas as pd

from torch.utils.data import Dataset
from datasets import load_dataset

def download_dataset(subset_size=None):
    if (not os.path.exists('../data')):
        os.makedirs('../data')
    train_ds = load_dataset("bookcorpus")
    train_ds = train_ds['train']
    if subset_size is not None:
        train_ds = train_ds.select(range(int(len(train_ds)*subset_size)))
    train_ds.to_csv('../data/bookcorpus.csv')

def tokenizeDataset(csv_file):
    df = pd.read_csv(csv_file)
    word2idx = dict()
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1

    for row in df['text']:
        for word in row.split():
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    
    with open('../data/word2idx.json', 'w') as f:
        json.dump(word2idx, f)
    
    idx2word = {v: k for k, v in word2idx.items()}
    with open('../data/idx2word.json', 'w') as f:
        json.dump(idx2word, f)
    
class BookCorpus():
    """ This dataset contains the sentences from the bookcorpus dataset
    """
    def __init__(self):
        self.df = pd.read_csv("../data/bookcorpus.csv")
        self.longest_sent = len(max(self.df['text'], key=len).split())
        self.word2idx = json.load(open('../data/word2idx.json', 'r'))
    
    def get_len(self):
        return len(self.df)

    def get_item(self, idx):
        sentence = self.df["text"].iloc[idx] 
        sentence = [self.word2idx[word] for word in sentence.split()] #Tokenize TODO add unk if word not in vocab
        sentence = sentence + [self.word2idx["<pad>"]]*(self.longest_sent - len(sentence)) #Pad
        return torch.tensor(sentence)

class CPCDataset(Dataset):
    """ This dataset contains the triplets (history, positive, negative) indexes for contrastive learning
    """
    def __init__(self, dataset, history_samples=10, prediction_len=3, negative_samples=10):
        self.dataset = dataset
        len_dataset = self.dataset.get_len() #Len of the dataset containing the sentences
        self.history_samples = history_samples
        self.prediction_len = prediction_len
        self.contrastive_dataset = pd.DataFrame(columns=['history','positive','negative', 'step'])
        for i in range(len_dataset):
            if i < len_dataset - history_samples - prediction_len:
                prediction_step = random.randint(1, prediction_len)
                history_idx = [i+j for j in range(history_samples)]
                positive_idx = i+history_samples+prediction_step
                negative_idx = [random.randint(0, len_dataset-1) for _ in range(negative_samples)]
                self.contrastive_dataset = pd.concat([self.contrastive_dataset, pd.DataFrame([[history_idx, positive_idx, negative_idx, prediction_step]], columns=['history','positive','negative', 'step'])])

    def __len__(self):
        return len(self.contrastive_dataset)

    def __getitem__(self, idx):
        row = self.contrastive_dataset.iloc[idx]
        history = torch.stack([self.dataset.get_item(i) for i in row['history']])
        positive = self.dataset.get_item(row['positive'])
        negative = torch.stack([self.dataset.get_item(i) for i in row['negative']])
        #create a tensor that contains [tensor(history), tensor(positive), tensor(negative), tensor(step)]
        #warning they are not of the same size
        return history, positive, negative, torch.tensor(row['step'])



if __name__ == "__main__":
    download_dataset(subset_size=0.01)
    tokenizeDataset('../data/bookcorpus.csv')
    #dataset = BookCorpus()
    #print(dataset.df.head())