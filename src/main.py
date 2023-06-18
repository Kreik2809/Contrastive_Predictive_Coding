import argparse
import json
import torch

import numpy as np
import pandas as pd

from data.data_loader import download_dataset, tokenizeDataset
from data.dataset import NLPDataset, CPCDataset

from model.models import CpcModel, SentenceEncoder
from model.trainer import train
from model.evaluate import evaluate

def main():
    #Read json config file from command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/config.json')
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))

    data_dir = config["data_dir"]
    model_dir = config["model_dir"]
    log_dir = config["log_dir"]
    output_dir = config["output_dir"]

    train_dataset = config["train_dataset"]
    test_dataset = config["test_dataset"]
    train_dataset_file = config["train_dataset_file"]
    test_dataset_file = config["test_dataset_file"]

    generate_cpc_dataset = False
    if config["download_data"]:
        print("Downloading data...")
        generate_cpc_dataset = True
        train_data_subset = config["train_data_subset"]
        test_data_subset = config["test_data_subset"]
        download_dataset(dataset_name=train_dataset, subset_size=train_data_subset, test=False, data_dir=data_dir, csv_file=train_dataset_file)
        download_dataset(dataset_name=test_dataset, subset_size=test_data_subset, test=True, data_dir=data_dir, csv_file=test_dataset_file)
        tokenizeDataset(dataset_csv_file=data_dir+train_dataset_file+"_train.csv", data_dir=data_dir, output_file_prefix=train_dataset_file)

    if config["train"]:
        print("Training...")
        device = torch.device(config["device"])
        torch.manual_seed(config["seed"])
        train_dataset = NLPDataset(data_dir, train_dataset_file+"_train.csv", train_dataset_file+"_word2idx.json", 132)
        cpc_dataset = CPCDataset(train_dataset, config["history_samples"], config["prediction_length"], config["negative_samples"], generate_cpc_dataset, data_dir, config["cpc_dataset_file"])
        vocab_size = len(train_dataset.word2idx)
        model = CpcModel(device, vocab_size, config["embedding_dim"], config["output_dim"], config["hidden_dim"])
        train(model, cpc_dataset, config["batch_size"], config["epochs"], config["patience"], config["learning_rate"], config["weight_decay"], log_dir, output_dir, model_dir, config["models_file_prefix"])
    
    if config["test"]:
        print("Testing...")
        train_dataset = NLPDataset(data_dir, test_dataset_file+"_train.csv", train_dataset_file+"_word2idx.json", 37, True)
        test_dataset = NLPDataset(data_dir, test_dataset_file+"_test.csv", train_dataset_file+"_word2idx.json", 17, True)

        train_df = pd.read_csv(data_dir+test_dataset_file+"_train.csv")
        test_df = pd.read_csv(data_dir+test_dataset_file+"_test.csv")

        for index, row in train_df.iterrows():
            train_df.at[index, "text"] = train_dataset.get_item(index+2)
        for index, row in test_df.iterrows():
            test_df.at[index, "text"] = test_dataset.get_item(index+2)

        result_df = pd.DataFrame(columns=["model_name", "train_accuracy", "test_accuracy"])

        vocab_size = len(train_dataset.word2idx)
        encoder = SentenceEncoder(vocab_size, config["embedding_dim"], config["output_dim"])
        def init_weights(m):
            for name, param in m.named_parameters():
                torch.nn.init.uniform_(param.data, -0.08, 0.08)
        encoder.apply(init_weights)
        result_df = evaluate(encoder, train_df, test_df, result_df, "random_encoder") 
        encoder.load_state_dict(torch.load(model_dir+config["models_file_prefix"]+"_encoder.pt", map_location=torch.device('cpu')))
        result_df = evaluate(encoder, train_df, test_df, result_df, "trained_encoder")
        result_df.to_markdown(output_dir+"result.md", index=False)
        result_df.style.to_latex(output_dir+"result.tex")

if __name__ == "__main__":
    main()