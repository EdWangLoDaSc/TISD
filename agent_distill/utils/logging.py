import os
import logging
import json
from typing import Dict, Any, Optional


logger = logging.getLogger("tisd")


def setup_logging(config: Dict[str, Any], iteration: Optional[int] = None):
    """Set up file + console logging, optionally wandb."""
    save_dir = config["logging"]["save_dir"]
    if iteration is not None:
        log_dir = os.path.join(save_dir, f"iter_{iteration}")
    else:
        log_dir = save_dir
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "train.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root = logging.getLogger("tisd")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Save config
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Logging to {log_file}")
    return log_dir


def setup_wandb(config: Dict[str, Any], iteration: Optional[int] = None):
    """Initialize wandb run if available."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping")
        return None

    project = config["logging"].get("wandb_project", "tisd-alfworld")
    run_name = f"iter_{iteration}" if iteration is not None else "tisd"
    run = wandb.init(project=project, name=run_name, config=config, reinit=True)
    return run


def log_metrics(metrics: Dict[str, float], step: int, prefix: str = ""):
    """Log metrics to logger and wandb if active."""
    msg_parts = [f"{prefix}/{k}: {v:.4f}" if prefix else f"{k}: {v:.4f}"
                 for k, v in metrics.items()]
    logger.info(f"Step {step} | " + " | ".join(msg_parts))

    try:
        import wandb
        if wandb.run is not None:
            log_dict = {f"{prefix}/{k}" if prefix else k: v
                        for k, v in metrics.items()}
            log_dict["step"] = step
            wandb.log(log_dict)
    except ImportError:
        pass
