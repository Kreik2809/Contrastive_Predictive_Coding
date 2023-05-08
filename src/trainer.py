import torch

import numpy as np

from dataset import *
from models import *
from tqdm import tqdm

import matplotlib.pyplot as plt


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    embedding_dim = 128
    output_dim = 2400
    hidden_dim = 2400
    batch_size = 2
    epochs = 50
    patience = 10
    history_len = 10
    negative_len = 10
    print("Starting")
    dataset = NLPDataset()
    print(dataset.get_len())
    cpc_dataset = CPCDataset(dataset)
    print("Dataset loaded")
    vocab_size = len(json.load(open('../data/word2idx.json', 'r')))
    
    model = CpcModel(device, vocab_size, embedding_dim, output_dim, hidden_dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_val_split = int(len(cpc_dataset) * 0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(cpc_dataset, [train_val_split, len(cpc_dataset) - train_val_split])

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss_hist = np.array([])
    val_loss_hist = np.array([])
    train_accuracy_hist = np.array([])
    val_accuracy_hist = np.array([])

    print("Start training")

    for epoch in tqdm(range(epochs)):
        loss_batch = np.array([])
        acc_batch = np.array([])
        for batch in train_data_loader:
            model.zero_grad()
            loss, train_accuracy = model(batch)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            loss_batch = np.append(loss_batch, [loss.to('cpu').detach().numpy()])
            acc_batch = np.append(acc_batch, [train_accuracy])

        with torch.no_grad():
            val_loss_batch = np.array([])
            val_acc_batch = np.array([])
            for batch in val_data_loader:
                val_loss, val_accuracy = model(batch)
                val_loss = val_loss.mean()
                val_loss_batch = np.append(val_loss_batch, [val_loss.to('cpu').detach().numpy()])
                val_acc_batch = np.append(val_acc_batch, [val_accuracy])
        
        train_loss_hist = np.append(train_loss_hist, loss_batch.mean())
        val_loss_hist = np.append(val_loss_hist, val_loss_batch.mean())
        train_accuracy_hist = np.append(train_accuracy_hist, acc_batch.mean())
        val_accuracy_hist = np.append(val_accuracy_hist, val_acc_batch.mean())
   
        if epoch == 0:
            patience_counter = 0
            best_model = model
            best_val_loss = val_loss_hist[-1]
            best_val_accuracy = val_accuracy_hist[-1]

        elif val_loss_batch.mean() > best_val_loss:
            patience_counter += 1
            if patience_counter == patience:
                break   
        
        else:
            patience_counter = 0
            best_model = model
            best_val_loss = val_loss_hist[-1]
            best_val_accuracy = val_accuracy_hist[-1]

        print("Epoch: {} Train Loss: {} Val Loss: {}".format(epoch, loss_batch.mean(), val_loss_batch.mean()))
        print("Epoch: {} Train Accuracy: {} Val Accuracy: {}".format(epoch, acc_batch.mean(), val_acc_batch.mean()))
        

    print("Saving model with val loss: {} and val accuracy: {}".format(best_val_loss, best_val_accuracy))
    best_model.save("output/model")

    plt.figure(figsize=(10,10))
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(val_loss_hist, label="Val Loss")
    plt.legend()
    plt.savefig("output/loss.png")

    plt.figure(figsize=(10,10))
    plt.plot(train_accuracy_hist, label="Train Accuracy")
    plt.plot(val_accuracy_hist, label="Val Accuracy")
    plt.legend()
    plt.savefig("output/accuracy.png")
        
if __name__ == "__main__":
    torch.manual_seed(42)
    train()