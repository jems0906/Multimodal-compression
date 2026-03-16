from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from pathlib import Path
import os

def load_config(config_path: Path) -> DictConfig:
    """Load configuration using Hydra."""
    config_dir = str(config_path.parent.resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_path.stem)
    return cfg