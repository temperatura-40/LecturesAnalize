"""Configuration loader for the lecture notes pipeline.

Reads config.json from the project root or user home.
Supports ~ expansion for paths.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _expand_path(path_str: str) -> str:
    """Expand ~ to home directory and resolve."""
    expanded = os.path.expanduser(path_str)
    return os.path.normpath(expanded)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to config file. Defaults to
            <cwd>/config.json or ~/.lecture_notes/config.json.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        # Look for config.json relative to the project root (where src/config.py lives)
        project_root = Path(__file__).resolve().parent.parent
        project_config = project_root / "config.json"
        if project_config.is_file():
            config_path = str(project_config)
        else:
            # Fallback: check user home
            home_config = str(Path.home() / ".lecture_notes" / "config.json")
            if Path(home_config).is_file():
                config_path = home_config
            else:
                config_path = str(project_config)  # Will raise if missing

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Expand all paths in folders
    folders = cfg.get("folders", {})
    for key in folders:
        if isinstance(folders[key], str):
            folders[key] = _expand_path(folders[key])

    # Ensure folders are updated
    cfg["folders"] = folders

    return cfg