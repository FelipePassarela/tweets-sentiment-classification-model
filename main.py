from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb
from src.datasets.text_dataset import TextDataset
from src.models.lstm_sentiment_classifier import LSTMSentimentClassifier
from src.training.eval import test
from src.training.train import train
from src.utils.getters import get_config


def main():
    config = get_config()
    
    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer"]["name"])
    # model = AutoModelForSequenceClassification.from_pretrained(config["data"]["tokenizer"]["name"])
    model = LSTMSentimentClassifier(vocab_size=tokenizer.vocab_size,
                                    embedding_dim= config["model"]["embedding_dim"],
                                    hidden_dim=    config["model"]["hidden_dim"],
                                    output_dim=    config["model"]["output_dim"],
                                    n_layers=      config["model"]["n_layers"],
                                    bidirectional= config["model"]["bidirectional"],
                                    dropout=       config["model"]["dropout"])

    model_name = model.__class__.__name__
    date = datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    wandb.init(project=config["wandb"]["project_name"], config=config, name=f"{model_name}{date}")

    sample_input = {"input_ids": torch.zeros((config["training"]["batch_size"], 
                                 config["data"]["tokenizer"]["max_length"]), 
                                 dtype=torch.long),
                    "attention_mask": torch.ones((config["training"]["batch_size"], 
                                      config["data"]["tokenizer"]["max_length"]), 
                                      dtype=torch.long)}
    summary(model, input_data=sample_input)

    train_dataset = TextDataset(config["data"]["processed"]["training"], tokenizer)
    val_dataset = TextDataset(config["data"]["processed"]["validation"], tokenizer)
    test_dataset = TextDataset(config["data"]["processed"]["test"], tokenizer)

    train_loader = DataLoader(train_dataset,
                              batch_size=config["training"]["batch_size"],
                              shuffle=True,
                              num_workers=config["training"]["num_workers"],
                              pin_memory=config["training"]["pin_memory"])
    val_loader = DataLoader(val_dataset,
                            batch_size=config["training"]["batch_size"],
                            shuffle=False,
                            num_workers=config["training"]["num_workers"],
                            pin_memory=config["training"]["pin_memory"])
    test_loader = DataLoader(test_dataset,
                             batch_size=config["training"]["batch_size"],
                             shuffle=False,
                             num_workers=config["training"]["num_workers"],
                             pin_memory=config["training"]["pin_memory"])

    train(model, train_loader, val_loader)
    test(model, test_loader)

    wandb.finish()


if __name__ == "__main__":
    main()