import kagglehub


if __name__ == "__main__":
    path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")
    print("Path to dataset files:", path)
