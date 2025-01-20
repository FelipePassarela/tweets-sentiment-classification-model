import pandas as pd
import pytest
import yaml
from transformers import AutoTokenizer

from src.datasets.text_dataset import TextDataset


@pytest.fixture
def config():
    with open("src/config/config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def tokenizer(config):
    tokenizer_name = config["data"]["tokenizer"]["name"]
    return AutoTokenizer.from_pretrained(tokenizer_name)


@pytest.fixture
def dataset(config, tokenizer):
    train_path = config["data"]["processed"]["training"]
    return TextDataset(train_path, tokenizer)


def test_dataset_item(dataset, config):
    item = dataset[0]
    max_length = config["data"]["tokenizer"]["max_length"]
    label_map = config["data"]["label_map"]
    
    assert "input_ids" in item, "Dataset item should contain input_ids"
    assert "attention_mask" in item, "Dataset item should contain attention_mask"
    assert "labels" in item, "Dataset item should contain labels"
    
    assert item["input_ids"].shape == (1, max_length), f"input_ids shape should be (1, {max_length})"
    assert item["attention_mask"].shape == (1, max_length), f"attention_mask shape should be (1, {max_length})"
    assert item["labels"] in label_map.values(), f"labels should be in {label_map.values()}"


def test_dataset_initialization(config, tokenizer):
    train_path = config["data"]["processed"]["training"]
    max_length = config["data"]["tokenizer"]["max_length"]
    
    dataset = TextDataset(train_path, tokenizer)
    assert isinstance(dataset.data, pd.DataFrame), "Dataset data should be a pandas DataFrame"
    assert dataset.max_length == max_length, "Dataset max_length should match config"
    assert dataset.tokenizer == tokenizer, "Dataset tokenizer should match input tokenizer"
