"""
Configuration utilities module.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration paths
DEFAULT_CONFIG_PATH = os.path.join("config", "settings.yaml")


def read_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Read configuration from a YAML or JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    path = config_path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)

    try:
        if not os.path.exists(path):
            logger.warning(f"Configuration file {path} not found")
            return {}

        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                try:
                    return yaml.safe_load(f)
                except yaml.YAMLError as e:
                    logger.error(f"Error parsing YAML config: {str(e)}")
                    return {}
            elif path.endswith('.json'):
                try:
                    return json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON config: {str(e)}")
                    return {}
            else:
                logger.warning(f"Unsupported config file format: {path}")
                return {}
    except Exception as e:
        logger.error(f"Error reading config file: {str(e)}")
        return {}


def update_config(updates: Dict[str, Any], config_path: Optional[str] = None) -> bool:
    """
    Update configuration settings.

    Args:
        updates: Settings to update
        config_path: Path to the configuration file

    Returns:
        True if successful, False otherwise
    """
    path = config_path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)

    try:
        # Read current config
        current_config = read_config(path)

        # Apply updates
        current_config.update(updates)

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Write updated config
        with open(path, 'w') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                yaml.dump(current_config, f, default_flow_style=False)
            elif path.endswith('.json'):
                json.dump(current_config, f, indent=2)
            else:
                logger.warning(f"Unsupported config file format: {path}")
                return False

        logger.info(f"Updated configuration in {path}")
        return True
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        return False