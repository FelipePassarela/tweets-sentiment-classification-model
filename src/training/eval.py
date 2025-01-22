from typing import Callable

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.utils.getters import get_config, get_loss_fn


def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: str,
    is_testing: bool = False,
) -> dict[str, float]:
    """
    Evaluate the model on a dataset.

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
    progress = tqdm(dataloader,
                    total=num_batches,
                    desc="Validation" if not is_testing else "Testing",
                    unit="batch",
                    colour="yellow" if not is_testing else "blue")

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

    model.eval()
    with torch.no_grad():
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            preds = model(**batch)
            preds = preds.logits if hasattr(preds, "logits") else preds
            loss = criterion(preds, labels)

            mean_loss.update(loss.item())
            running_loss.update(loss.item())
            metrics.update(preds, labels)
            computed_metrics = metrics.compute()

            progress.set_postfix({
                "Loss": f"{running_loss.compute():.4f}",
                "Accuracy": f"{computed_metrics["MulticlassAccuracy"].item():.4f}",
                "ROC AUC": f"{computed_metrics["MulticlassAUROC"].item():.4f}"})

    avg_metrics = {"Loss": mean_loss.compute(), **computed_metrics}
    return avg_metrics


def test(model: nn.Module, test_loader: DataLoader):
    """
    Test a trained model on the test dataset and logs metrics to wandb.
    
    The function:
    - Uses the configured loss function
    - Evaluates model on test data using eval_one_epoch()
    - Logs test metrics (Loss, Accuracy, AUROC) to wandb
    - Prints key test metrics to console

    Args:
        model (nn.Module): The neural network model to test
        test_loader (DataLoader): DataLoader containing the test dataset

    Returns:
        None
    """
    config = get_config()

    criterion = get_loss_fn(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_metrics = eval_one_epoch(model=model,
                                  dataloader=test_loader,
                                  criterion=criterion,
                                  device=device,
                                  is_testing=True)

    test_metrics = {f"Test/{k.replace("Multiclass", "")}": v for k, v in test_metrics.items()}
    wandb.log(test_metrics)
    print(f"Test Loss: {test_metrics['Test/Loss']:.4f} | "
          f"Test Accuracy: {test_metrics['Test/Accuracy']:.4f} | "
          f"Test ROC AUC: {test_metrics['Test/AUROC']:.4f}")
