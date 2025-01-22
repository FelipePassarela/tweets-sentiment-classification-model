import os

import torch
import torch.nn as nn


def load_model(model: nn.Module, path: str, device: str = None) -> nn.Module:
    """
    Load a PyTorch model from a file.

    Args:
        model (nn.Module): The PyTorch model to load the state dictionary into
        path (str): The path to the file
        device (str): The device to load the model onto (default: None)

    Returns:
        nn.Module: The model with the state dictionary loaded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
        
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    return model


def save_model(model: nn.Module, path: str):
    """
    Save a PyTorch model to a file.

    Args:
        model (nn.Module): The PyTorch model to save
        path (str): The path to the file

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)