import torch.optim as optim


def get_optimizer(model, config):
    name = config["training"]["optimizer"]
    learning_rate = config["training"]["learning_rate"]

    match name:
        case "adam":
            return optim.Adam(model.parameters(), lr=learning_rate)
        case "adamw":
            return optim.AdamW(model.parameters(), lr=learning_rate)
        case "sgd":
            return optim.SGD(model.parameters(), lr=learning_rate)
        case _:
            raise ValueError(f"Unknown optimizer: {name}")
