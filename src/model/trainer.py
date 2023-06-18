import torch
import datetime
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils as utils

from tqdm import tqdm

def train(model, cpc_dataset, batch_size, epochs, patience, lr, weight_decay, log_dir, output_dir, models_dir, models_file_prefix):
    """ Train the CPC model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_dataset = torch.utils.data.Subset(cpc_dataset, range(0, int(len(cpc_dataset)*0.8)))
    val_dataset = torch.utils.data.Subset(cpc_dataset, range(int(len(cpc_dataset)*0.8), len(cpc_dataset))) 

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss_hist = np.array([])
    val_loss_hist = np.array([])
    train_accuracy_hist = np.array([])
    val_accuracy_hist = np.array([])

    log_file = log_dir + "train_log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info("Hyperparameters:")
    logger.info("Batch size: " + str(batch_size))
    logger.info("Epochs: " + str(epochs))
    logger.info("Patience: " + str(patience))
    logger.info("Learning rate: " + str(lr))
    logger.info("Weight decay: " + str(weight_decay))
    logger.info("Embedding dim: " + str(model.embedding_dim))
    logger.info("Output dim: " + str(model.output_dim))
    logger.info("Hidden dim: " + str(model.hidden_dim))

    logger.info("Start training...")
    print("Start training...")
    for epoch in tqdm(range(epochs)):
        loss_batch = np.array([])
        acc_batch = np.array([])
        for batch in train_data_loader:
            model.zero_grad()
            loss, train_accuracy = model(batch)
            loss = loss.mean()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss_batch = np.append(loss_batch, [loss.to('cpu').detach().numpy()])
            acc_batch = np.append(acc_batch, [train_accuracy])

            del loss
        
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

        logger.info("Epoch: {} Train Loss: {} Val Loss: {} Train Accuracy: {} Val Accuracy: {}".format(epoch, loss_batch.mean(), val_loss_batch.mean(), acc_batch.mean(), val_acc_batch.mean()))
        
    logger.info("End of training")
    best_model = model #Val loss is never decreasing because of insuficient data, take the last model anyway
    best_val_loss = val_loss_hist[-1]
    best_val_accuracy = val_accuracy_hist[-1]
    logger.info("Saving model with val loss: {} and val accuracy: {}".format(best_val_loss, best_val_accuracy))
    best_model.save(models_dir, models_file_prefix)

    plt.figure(figsize=(10,10))
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(val_loss_hist, label="Val Loss")
    plt.legend()
    plt.savefig(output_dir + "loss.png")

    plt.figure(figsize=(10,10))
    plt.plot(train_accuracy_hist, label="Train Accuracy")
    plt.plot(val_accuracy_hist, label="Val Accuracy")
    plt.legend()
    plt.savefig(output_dir + "accuracy.png")