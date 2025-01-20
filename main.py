import torch
from torchinfo import summary
from transformers import AutoModelForSequenceClassification

from src.config.config_loader import get_config
from src.training.train import train


def main():
    config = get_config()
    model = AutoModelForSequenceClassification.from_pretrained(config["model"]["name"])
    sample_input = {
        "input_ids": torch.zeros((config["training"]["batch_size"], 
                                  config["data"]["tokenizer"]["max_length"]), 
                                  dtype=torch.long),
        "attention_mask": torch.ones((config["training"]["batch_size"], 
                                      config["data"]["tokenizer"]["max_length"]), 
                                      dtype=torch.long)}
    summary(model, input_data=sample_input)
    train(model)


if __name__ == "__main__":
    main()