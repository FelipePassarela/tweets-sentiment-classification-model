import numpy as np
import torch.nn as nn

from src.utils.model_loader import save_model


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0, mode: str = "min"):
        """
        Initialize the EarlyStopping class.

        This class implements early stopping mechanism for training neural networks.
        It monitors a given metric and stops training when no improvement is seen
        for a specified number of epochs.

        Args:
            patience (int): Number of epochs to wait before early stopping.
            min_delta (float, optional): Minimum change in monitored quantity to qualify
                as an improvement. Defaults to 0.
            mode (str, optional): One of {"min", "max"}. In 'min' mode, training stops
                when quantity monitored has stopped decreasing; in 'max' mode it will
                stop when the quantity monitored has stopped increasing. Defaults to "min".

        Attributes:
            counter (int): Counter for epochs without improvement.
            best_score (float): Best score observed.
            early_stop (bool): Flag indicating if early stopping criteria is met
            val_loss_min (float): Minimum validation loss observed (for "min" mode) or
                maximum (for "max" mode).
        """

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf if mode == "min" else -np.Inf

    def __call__(self, score: float, model: nn.Module, path: str):
        """
        Called when monitoring a metric for early stopping training.

        This method compares the current score with the best score seen so far and decides 
        whether to save the model checkpoint or increment the early stopping counter.

        Args:
            score (float): Current validation metric to monitor
            model (nn.Module): PyTorch model to save if score improves
            path (str): Path where model checkpoint will be saved

        Returns:
            None

        Side Effects:
            - Updates self.best_score if current score is better
            - Saves model checkpoint if score improves
            - Increments early stopping counter if score doesn't improve
            - Sets early_stop flag to True if counter exceeds patience
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, path)
        else:
            if self.mode == "min":
                condition = score > self.best_score + self.min_delta
            else:
                condition = score < self.best_score - self.min_delta

            if condition:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(score, model, path)
                self.counter = 0

    def save_checkpoint(self, score: float, model: nn.Module, path: str):
        """
        Saves a checkpoint of the model if the validation score improves.

        Args:
            score (float): The current validation score to compare against the minimum.
            model (nn.Module): The PyTorch model to save.
            path (str): File path where the model checkpoint will be saved.

        Notes:
            - Creates necessary directories if they don't exist
            - Updates the minimum validation loss (self.val_loss_min)
            - Saves only the model's state dictionary
        """
        save_model(model, path)
        self.val_loss_min = score
