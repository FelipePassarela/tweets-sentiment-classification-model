import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.getters import get_config


def preprocess(data_path: str, output_path: str):
    """
    Preprocesses the raw data by mapping sentiment labels and saving to a new CSV file.

    This function reads a CSV file, drops any rows with missing values, maps sentiment
    labels according to the configuration, and saves the processed data to a new file.

    Args:
        data_path (str): Path to the input CSV file containing raw data.
        output_path (str): Path where the preprocessed CSV file will be saved.

    Returns:
        None

    Notes:
        - Input CSV should have columns: "id", "entity", "sentiment", "text"
        - Sentiment labels are mapped according to label_map in config
        - Creates output directory if it doesn't exist
        - Prints confirmation message when save is complete
    """
    config = get_config()

    columns = ["id", "entity", "sentiment", "text"]
    label_map = config["data"]["label_map"]
    
    data = pd.read_csv(data_path, names=columns)
    data = data.dropna()
    data["sentiment"] = data["sentiment"].map(label_map)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    config = get_config()

    preprocess(config["data"]["raw"]["training"],
               config["data"]["processed"]["training"])
    preprocess(config["data"]["raw"]["test"],
               config["data"]["processed"]["test"])

    train_size = config["data"]["train_val_split"]
    df_train = pd.read_csv(config["data"]["processed"]["training"])
    df_train, df_val = train_test_split(df_train, train_size=train_size, random_state=42)
    df_train.to_csv(config["data"]["processed"]["training"], index=False)
    df_val.to_csv(config["data"]["processed"]["validation"], index=False)
    print(f"Preprocessing data saved to {config['data']['processed']['validation']}") 
