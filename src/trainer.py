import torch
import numpy as np

from dataset import *
from models import *

import matplotlib.pyplot as plt


def train():
    dataset = BookCorpus()
    dataset.df = dataset.df[:100]
    
    cpc_dataset = CPCDataset(dataset)

    vocab_size = len(json.load(open('../data/word2idx.json', 'r')))
    embedding_dim = 128
    output_dim = 2400
    hidden_dim = 2400
    batch_size = 64
    epochs = 2
    model = CpcModel(vocab_size, embedding_dim, output_dim, hidden_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    data_loader = torch.utils.data.DataLoader(cpc_dataset, batch_size=batch_size, shuffle=True)
    
    loss_hist = np.array([])
    for epoch in range(epochs):
        for batch in data_loader:
            model.zero_grad()
            loss = model(batch).mean()
            loss.backward()
            optimizer.step()
            print(loss)
            loss_hist = np.append(loss_hist, [loss.detach().numpy()])
    
    plt.figure(figsize=(10,10))
    plt.plot(loss_hist)
    plt.savefig("loss.png")
        

if __name__ == "__main__":
    torch.manual_seed(42)
    train()