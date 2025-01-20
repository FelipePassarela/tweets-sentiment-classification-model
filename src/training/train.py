from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb
from src.config.config_loader import get_config
from src.datasets.text_dataset import TextDataset
from src.utils.optimizers import get_optimizer


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: Callable,
    device: str,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    """
    Train the model for one epoch.

    This function performs one training epoch, processing all batches in the dataloader
    and updating the model's parameters using the specified optimizer and loss criterion.
    It also computes and tracks various metrics during training.

    Args:
        model (nn.Module): The neural network model to train
        dataloader (DataLoader): DataLoader containing the training data
        optimizer (optim.Optimizer): The optimizer used for updating model parameters
        criterion (Callable): The loss function used for computing training loss
        device (str): Device to run the training on ('cuda' or 'cpu')
        scaler (torch.amp.GradScaler): Gradient scaler for mixed precision training

    Returns:
        dict[str, float]: Dictionary containing average metrics for the epoch including:
            - loss: Average training loss
            - accuracy: Classification accuracy
            - f1_score: F1 score
            - precision: Precision score
            - recall: Recall score
            - auroc: Area Under ROC curve
            - confusion_matrix: Confusion matrix for predictions
    """
    num_batches = len(dataloader)
    progress = tqdm(
        enumerate(dataloader),
        total=num_batches,
        desc="Training",
        unit="batch",
        colour="green"
    )

    num_classes = get_config()["data"]["num_classes"]
    metrics = torchmetrics.MetricCollection([
        torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        torchmetrics.F1Score(task="multiclass", num_classes=num_classes),
        torchmetrics.Precision(task="multiclass", num_classes=num_classes),
        torchmetrics.Recall(task="multiclass", num_classes=num_classes),
        torchmetrics.AUROC(task="multiclass", num_classes=num_classes),
        torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes),
    ]).to(device)
    running_loss = 0.0

    model.train()
    for i, batch in progress:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        
        with torch.autocast(device):
            preds = model(**batch).logits
            loss = criterion(preds, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        metrics.update(preds, labels)
        computed_metrics = metrics.compute()

        progress.set_postfix({
            "Loss": f"{running_loss / (i + 1):.4f}",
            "Accuracy": f"{computed_metrics['MulticlassAccuracy'].item():.4f}",
            "ROC AUC": f"{computed_metrics['MulticlassAUROC'].item():.4f}",
        })

    avg_metrics = {"Loss": running_loss / num_batches, **computed_metrics}
    return avg_metrics


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: str,
    is_testing: bool = False,
) -> dict[str, float]:
    """
    Validates or tests the model for one epoch using the provided dataloader.

    This function evaluates the model's performance by computing various metrics including
    accuracy, F1 score, precision, recall, AUROC and confusion matrix. It processes the 
    data in batches and displays a progress bar with live metrics.

    Args:
        model (nn.Module): PyTorch model to validate/test
        dataloader (DataLoader): DataLoader containing validation/test data
        criterion (Callable): Loss function to calculate model error
        device (str): Device to run the validation on ('cpu' or 'cuda')
        is_testing (bool, optional): Flag to indicate if this is a testing run. 
                                     Affects progress bar appearance. Defaults to False.

    Returns:
        dict[str, float]: Dictionary containing averaged metrics including:
            - loss: Average loss over all batches
            - accuracy: Model accuracy
            - f1_score: F1 score
            - precision: Precision score
            - recall: Recall score
            - auroc: Area Under the ROC Curve
            - confusion_matrix: Confusion matrix for multiclass classification

    Note:
        The function runs in evaluation mode and doesn't compute gradients.
    """
    num_batches = len(dataloader)
    progress = tqdm(
        enumerate(dataloader), 
        total=num_batches, 
        desc="Validation" if not is_testing else "Testing",
        unit="batch", 
        colour="yellow" if not is_testing else "blue"
    )

    num_classes = get_config()["data"]["num_classes"]
    metrics = torchmetrics.MetricCollection([
        torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        torchmetrics.F1Score(task="multiclass", num_classes=num_classes),
        torchmetrics.Precision(task="multiclass", num_classes=num_classes),
        torchmetrics.Recall(task="multiclass", num_classes=num_classes),
        torchmetrics.AUROC(task="multiclass", num_classes=num_classes),
        torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes),
    ]).to(device)
    running_loss = 0.0

    model.eval()
    with torch.no_grad():
        for i, batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            preds = model(**batch).logits
            loss = criterion(preds, labels)

            running_loss += loss.item()
            metrics.update(preds, labels)
            computed_metrics = metrics.compute()

            progress.set_postfix({
                "Loss": f"{running_loss / (i + 1):.4f}",
                "Accuracy": f"{computed_metrics["MulticlassAccuracy"].item():.4f}",
                "ROC AUC": f"{computed_metrics["MulticlassAUROC"].item():.4f}",
            })

    avg_metrics = {"Loss": running_loss / num_batches, **computed_metrics}
    return avg_metrics


def train(model: nn.Module):
    """
    Train a PyTorch model for sentiment analysis.

    This function handles the complete training pipeline including:
    - Setting up device (CPU/GPU)
    - Loading and preparing datasets
    - Creating data loaders
    - Training for specified number of epochs
    - Validating after each epoch
    - Testing final model performance
    - Logging metrics to Weights & Biases

    Args:
        model (nn.Module): PyTorch model to be trained
        
    Returns:
        None
        
    The function expects a configuration file with the following structure:
        - data:
            - tokenizer:
                - name: Name of the pretrained tokenizer
            - processed:
                - training: Path to training data
                - validation: Path to validation data 
                - test: Path to test data
        - training:
            - batch_size: Batch size for training
            - num_workers: Number of workers for data loading
            - pin_memory: Whether to pin memory for data loading
            - epochs: Number of training epochs
        - wandb:
            - project_name: Name of the W&B project
    """
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer"]["name"])
    model = model.to(device)

    train_dataset = TextDataset(
        data_path=config["data"]["processed"]["training"],
        tokenizer=tokenizer,
    )
    val_dataset = TextDataset(
        data_path=config["data"]["processed"]["validation"],
        tokenizer=tokenizer,
    )
    test_dataset = TextDataset(
        data_path=config["data"]["processed"]["test"],
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    scaler = torch.amp.GradScaler()

    wandb.init(project=config["wandb"]["project_name"], config=config)

    num_epochs = config["training"]["epochs"]
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )
        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        train_metrics = {f"Train/{k.replace("Multiclass", "")}": v for k, v in train_metrics.items()}
        val_metrics = {f"Val/{k.replace("Multiclass", "")}": v for k, v in val_metrics.items()}
        wandb.log({**train_metrics, **val_metrics})
        
        print(f"Train Loss: {train_metrics['Train/Loss']:.4f} | "
              f"Train Accuracy: {train_metrics['Train/Accuracy']:.4f} | "
              f"Train ROC AUC: {train_metrics['Train/AUROC']:.4f}")
        print(f"Val Loss: {val_metrics['Val/Loss']:.4f} | "
              f"Val Accuracy: {val_metrics['Val/Accuracy']:.4f} | "
              f"Val ROC AUC: {val_metrics['Val/AUROC']:.4f}")
        
    test_metrics = validate_one_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        is_testing=True,
    )

    test_metrics = {f"Test/{k.replace("Multiclass", "")}": v for k, v in test_metrics.items()}
    wandb.log(test_metrics)

    print(f"Test Loss: {test_metrics['Test/Loss']:.4f} | "
          f"Test Accuracy: {test_metrics['Test/Accuracy']:.4f} | "
          f"Test ROC AUC: {test_metrics['Test/AUROC']:.4f}")

    wandb.finish()
    print("Training complete!")
