"""
Configuration package.

Exports the settings module and utilities for easy importing.
"""

from src.config.settings import settings
from src.config.utils import read_config, update_config

__all__ = ["settings", "read_config", "update_config"]

