import logging
import os
from typing import Optional

def setup_logger(name: str, log_level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Sets up a standardized logger."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def init_wandb(project_name: str, config: dict, run_name: Optional[str] = None):
    """Initializes Weights & Biases for experiment tracking."""
    try:
        import wandb
        wandb.init(project=project_name, config=config, name=run_name)
    except ImportError:
        logging.warning("wandb is not installed. Run `pip install wandb` to enable experiment tracking.")
