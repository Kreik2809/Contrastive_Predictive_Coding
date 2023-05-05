import os
import random
import json
import csv
import linecache
import torch

import pandas as pd

from torch.utils.data import Dataset
from datasets import load_dataset

def download_dataset(dataset_name="bookcorpus", subset_size=None, test=True):
    if (not os.path.exists('../data')):
        os.makedirs('../data')
    ds = load_dataset(dataset_name)
    train_ds = ds['train']
    if subset_size is not None:
        train_ds = train_ds.select(range(int(len(train_ds)*subset_size)))
        train_ds.to_csv('../data/'+dataset_name+'_train_reduce.csv', index=False)
    else:
        train_ds.to_csv('../data/'+dataset_name+'_train.csv', index=False)
    if test:
        test_ds = ds['test']
        test_ds.to_csv('../data/'+dataset_name+'_test.csv', index=False)

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
    
class NLPDataset():
    """ This dataset contains the sentences from the bookcorpus dataset
    """
    def __init__(self, csv_file="../data/bookcorpus_train_reduce.csv"):
        self.csv_file = csv_file
        self.longest_sent = 132 #TODO : find a way to get this value automatically
        self.word2idx = json.load(open('../data/word2idx.json', 'r'))
    
    def get_len(self):
        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file)
            row_count = sum(1 for row in reader)
        return row_count
               
    def get_item(self, idx):
        with open(self.csv_file, 'r') as file:
            line = linecache.getline(self.csv_file, idx)
            reader = csv.reader([line])
            row = next(reader)
        sentence = row[0]
        sentence = sentence.lower()
        sentence = [self.word2idx[word] if word in self.word2idx else self.word2idx["<unk>"] for word in sentence.split()] #Tokenize
        sentence = sentence + [self.word2idx["<pad>"]]*(self.longest_sent - len(sentence)) #Pad
        return torch.tensor(sentence)

class CPCDataset(Dataset):
    """ This dataset contains the triplets (history, positive, negative) indexes for contrastive learning
    """
    def __init__(self, dataset, learning_sample_len, history_samples=10, prediction_len=3, negative_samples=10):
        self.dataset = dataset
        self.learning_sample_len = learning_sample_len
        len_dataset = self.dataset.get_len() #Len of the dataset containing the sentences
        self.history_samples = history_samples
        self.prediction_len = prediction_len
        self.contrastive_dataset = pd.DataFrame(columns=['history','positive','negative', 'step'])
        for _ in range(self.learning_sample_len):
            t = random.randint(2, len_dataset - history_samples - prediction_len)
            prediction_step = random.randint(1, prediction_len)
            history_idx = [t+j for j in range(history_samples)]
            positive_idx = t+history_samples+prediction_step-1
            possible_negs = list(set(range(2, len_dataset)) - set(history_idx) - set([positive_idx]))
            negative_idx = random.sample(possible_negs, negative_samples)
            self.contrastive_dataset = pd.concat([self.contrastive_dataset, pd.DataFrame([[history_idx, positive_idx, negative_idx, prediction_step]], columns=['history','positive','negative', 'step'])])

    def __len__(self):
        return len(self.contrastive_dataset)

    def __getitem__(self, idx):
        row = self.contrastive_dataset.iloc[idx]
        history = torch.stack([self.dataset.get_item(i) for i in row['history']])
        positive = self.dataset.get_item(row['positive'])
        negative = torch.stack([self.dataset.get_item(i) for i in row['negative']])
        return history, positive, negative, torch.tensor(row['step'])


if __name__ == "__main__":
    #download_dataset("trec")
    #download_dataset("bookcorpus", subset_size = 0.1, test=False)
    #tokenizeDataset("../data/bookcorpus_train.csv")

    """
    print("Starting")
    dataset = NLPDataset()
    print(dataset.get_len())
    cpc_dataset = CPCDataset(dataset, 1)
    print(cpc_dataset.contrastive_dataset.iloc[0])
    print(cpc_dataset[0])
    print("Done")
    """