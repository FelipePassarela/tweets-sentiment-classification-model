import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.utils.getters import get_config


class TextDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer):
        super().__init__()

        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = get_config()["data"]["tokenizer"]["max_length"]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row["text"]
        entity = row["entity"]
        sep = self.tokenizer.sep_token
        combined = f"{entity}{sep}{text}"

        inputs = self.tokenizer(combined, 
                                max_length=self.max_length, 
                                padding="max_length", 
                                truncation=True, 
                                return_tensors="pt",
                                return_token_type_ids=False)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = row["sentiment"]
        return inputs

    def __len__(self):
        return len(self.data)
    