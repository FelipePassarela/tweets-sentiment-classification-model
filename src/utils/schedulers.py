from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR


def get_scheduler(optimizer: Optimizer, config: dict, steps_per_epoch: int):
    """
    Returns a PyTorch learning rate scheduler based on configuration parameters.

    This function creates and returns a learning rate scheduler for training deep learning models.
    It supports three types of schedulers: OneCycleLR, ReduceLROnPlateau, and CosineAnnealingLR.

    Args:
        optimizer (Optimizer): PyTorch optimizer to be used with the scheduler
        config (dict): Configuration dictionary containing training parameters
            Expected structure:
            {
                "training": {
                    "epochs": int,
                    "scheduler": {
                        "name": str,  # "one_cycle", "reduce_on_plateau", or "cosine"
                        "params": {  # Required for one_cycle scheduler
                            "max_lr": float,
                            "pct_start": float,
                            "div_factor": float,
                            "final_div_factor": float
                        }
                    }
                }
            }
        steps_per_epoch (int): Number of training steps per epoch

    Returns:
        torch.optim.lr_scheduler: One of the following schedulers:
            - OneCycleLR
            - ReduceLROnPlateau
            - CosineAnnealingLR

    Raises:
        ValueError: If an unknown scheduler name is provided in the config
    """
    scheduler_config = config["training"]["scheduler"]
    total_steps = config["training"]["epochs"] * steps_per_epoch
    
    match scheduler_config["name"]:
        case "one_cycle":
            return OneCycleLR(
                optimizer,
                max_lr=scheduler_config["params"]["max_lr"],
                total_steps=total_steps,
                pct_start=scheduler_config["params"]["pct_start"],
                div_factor=scheduler_config["params"]["div_factor"],
                final_div_factor=scheduler_config["params"]["final_div_factor"]
            )
        case "reduce_on_plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=2,
                verbose=True
            )
        case "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=0
            )
        case _:
            raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")
