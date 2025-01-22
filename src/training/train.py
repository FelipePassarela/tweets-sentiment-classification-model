from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.training.eval import eval_one_epoch
from src.utils.early_stopping import EarlyStopping
from src.utils.getters import get_config, get_loss_fn, get_scheduler
from src.utils.model_loader import load_model
from src.utils.optimizers import get_optimizer


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: object,
    criterion: _Loss,
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
        scheduler (object): Learning rate scheduler
        criterion (_Loss): The loss function used for computing training loss
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
            - learning_rate: Current learning rate
    """
    num_batches = len(dataloader)
    progress = tqdm(dataloader,
                    total=num_batches,
                    desc="Training",
                    unit="batch",
                    colour="green")

    num_classes = get_config()["data"]["num_classes"]
    metrics = torchmetrics.MetricCollection([
        torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        torchmetrics.F1Score(task="multiclass", num_classes=num_classes),
        torchmetrics.Precision(task="multiclass", num_classes=num_classes),
        torchmetrics.Recall(task="multiclass", num_classes=num_classes),
        torchmetrics.AUROC(task="multiclass", num_classes=num_classes),
        torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)]).to(device)
    running_loss = torchmetrics.RunningMean(window=100).to(device)
    mean_loss = torchmetrics.MeanMetric().to(device)

    model.train()
    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        
        with torch.autocast(device):
            preds = model(**batch)
            preds = preds.logits if hasattr(preds, "logits") else preds
            loss = criterion(preds, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()
            
        current_lr = optimizer.param_groups[0]['lr']
        running_loss.update(loss.item())
        mean_loss.update(loss.item())
        metrics.update(preds, labels)
        computed_metrics = metrics.compute()

        progress.set_postfix({
            "Loss": f"{running_loss.compute():.4f}",
            "Accuracy": f"{computed_metrics['MulticlassAccuracy'].item():.4f}",
            "ROC AUC": f"{computed_metrics['MulticlassAUROC'].item():.4f}",
            "LR": f"{current_lr:.2e}"})

    avg_metrics = {
        "Loss": mean_loss.compute(),
        "Learning Rate": current_lr,
        **computed_metrics}
    return avg_metrics


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
    """
    Train a PyTorch model for sentiment analysis.

    This function trains the model for a specified number of epochs, evaluating the model
    on the validation set after each epoch. It uses the specified optimizer and learning rate
    scheduler in configuration file to update the model's parameters.

    Args:
        model (nn.Module): PyTorch model to be trained

    Returns:
        None
    """
    config = get_config()
    model_name = type(model).__name__    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Training on {device}")

    criterion = get_loss_fn(config)
    optimizer = get_optimizer(model, config)
    scaler = torch.amp.GradScaler()
    scheduler = get_scheduler(optimizer=optimizer,
                              config=config,
                              steps_per_epoch=len(train_loader))
    early_stopping = EarlyStopping(patience=  config["training"]["early_stopping"]["patience"],
                                   min_delta= config["training"]["early_stopping"]["min_delta"],
                                   mode=      config["training"]["early_stopping"]["mode"])

    checkpoint_path = Path(config["training"]["early_stopping"]["checkpoint_path"]) / f"{model_name}.pt"
    num_epochs = config["training"]["epochs"]

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} - {model_name}")

        train_metrics = train_one_epoch(model=model,
                                        dataloader=train_loader,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        device=device,
                                        scaler=scaler)
        val_metrics = eval_one_epoch(model=model,
                                     dataloader=val_loader,
                                     criterion=criterion,
                                     device=device)

        train_metrics = {f"Train/{k.replace("Multiclass", "")}": v for k, v in train_metrics.items()}
        val_metrics = {f"Val/{k.replace("Multiclass", "")}": v for k, v in val_metrics.items()}
        wandb.log({**train_metrics, **val_metrics})
        
        print(f"Train Loss: {train_metrics['Train/Loss']:.4f} | "
              f"Train Accuracy: {train_metrics['Train/Accuracy']:.4f} | "
              f"Train ROC AUC: {train_metrics['Train/AUROC']:.4f}")
        print(f"Val Loss: {val_metrics['Val/Loss']:.4f} | "
              f"Val Accuracy: {val_metrics['Val/Accuracy']:.4f} | "
              f"Val ROC AUC: {val_metrics['Val/AUROC']:.4f}")
    
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_metrics["Loss"])

        early_stopping(val_metrics["Val/Loss"], model, checkpoint_path)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model = load_model(model, checkpoint_path, device=device)
    print("Training complete!")
