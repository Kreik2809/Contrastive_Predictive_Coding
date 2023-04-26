import json
import torch

import pandas as pd

from torch.utils.data import Dataset
from datasets import load_dataset

def download_dataset():
    train_ds = load_dataset("bookcorpus")
    train_ds = train_ds['train']
    train_ds.to_csv('../data/bookcorpus.csv', index=False)

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
    
class BookCorpus(Dataset):

    def __init__(self):
        self.df = pd.read_csv("../data/bookcorpus.csv")
        self.longest_sent = len(max(self.df['text'], key=len).split())
        self.word2idx = json.load(open('../data/word2idx.json', 'r'))
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence = self.df.iloc[idx] 
        sentence = [self.word2idx[word] for word in sentence.split()] #Tokenize TODO add unk if word not in vocab
        sentence = sentence + [self.word2idx.get(0)]*(self.longest_sent - len(sentence)) #Pad
        return torch.tensor(sentence)

if __name__ == "__main__":
    #download_dataset()
    #tokenizeDataset('../data/bookcorpus.csv')
    dataset = BookCorpus()
    print(dataset.df.head())