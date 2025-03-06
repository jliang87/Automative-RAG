"""
Logging utility functions for the Automotive Specs RAG system.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Union


def setup_logger(
    name: str = "automotive_rag",
    level: Union[int, str] = "INFO",
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Name of the logger
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, no file logging)
        log_to_console: Whether to log to console
        log_format: Custom log format (if None, use default format)
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging level if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Default log format
    if log_format is None:
        log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Add file handler if log_file provided
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if log_to_console is True
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


class GPULogger:
    """
    Logger for GPU operations with performance tracking.
    """
    
    def __init__(
        self,
        name: str = "gpu_operations",
        log_file: Optional[str] = None,
        log_level: Union[int, str] = "INFO",
    ):
        """
        Initialize GPU logger.
        
        Args:
            name: Name of the logger
            log_file: Path to log file (if None, use default)
            log_level: Logging level
        """
        if log_file is None:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = os.path.join(log_dir, f"gpu_{date_str}.log")
        
        self.logger = setup_logger(
            name=name,
            level=log_level,
            log_file=log_file,
            log_to_console=True,
        )
        
        # Try to log GPU information
        self._log_gpu_info()
    
    def _log_gpu_info(self):
        """Log information about available GPUs."""
        try:
            import torch
            
            self.logger.info(f"PyTorch version: {torch.__version__}")
            self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                self.logger.info(f"CUDA version: {torch.version.cuda}")
                self.logger.info(f"GPU count: {torch.cuda.device_count()}")
                
                for i in range(torch.cuda.device_count()):
                    self.logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    
                    # Try to get memory info
                    try:
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                        memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                        self.logger.info(f"  Memory allocated: {memory_allocated:.2f} GB")
                        self.logger.info(f"  Memory reserved: {memory_reserved:.2f} GB")
                    except:
                        self.logger.warning(f"  Unable to get memory info for GPU {i}")
        except ImportError:
            self.logger.warning("PyTorch not available, skipping GPU info logging")
        except Exception as e:
            self.logger.warning(f"Error logging GPU info: {str(e)}")
    
    def log_operation_start(self, operation: str, **kwargs):
        """
        Log the start of a GPU operation.
        
        Args:
            operation: Name of the operation
            **kwargs: Additional information to log
        """
        # Log start of operation
        log_message = f"Starting {operation}"
        
        # Add additional info if provided
        if kwargs:
            additional_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            log_message += f" ({additional_info})"
        
        self.logger.info(log_message)
        
        # Try to log current GPU memory usage
        try:
            import torch
            if torch.cuda.is_available():
                device_id = kwargs.get("device_id", 0)
                memory_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
                self.logger.info(f"GPU memory before operation: {memory_allocated:.2f} GB")
        except:
            pass
    
    def log_operation_end(self, operation: str, duration: float, success: bool = True, **kwargs):
        """
        Log the end of a GPU operation.
        
        Args:
            operation: Name of the operation
            duration: Duration of the operation in seconds
            success: Whether the operation was successful
            **kwargs: Additional information to log
        """
        # Format duration
        if duration < 1:
            duration_str = f"{duration * 1000:.2f} ms"
        else:
            duration_str = f"{duration:.2f} s"
        
        # Log status
        status = "completed" if success else "failed"
        log_message = f"{operation} {status} in {duration_str}"
        
        # Add additional info if provided
        if kwargs:
            additional_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            log_message += f" ({additional_info})"
        
        log_func = self.logger.info if success else self.logger.error
        log_func(log_message)
        
        # Try to log current GPU memory usage
        try:
            import torch
            if torch.cuda.is_available():
                device_id = kwargs.get("device_id", 0)
                memory_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
                self.logger.info(f"GPU memory after operation: {memory_allocated:.2f} GB")
        except:
            pass
    
    def log_error(self, operation: str, error: Exception, **kwargs):
        """
        Log an error during a GPU operation.
        
        Args:
            operation: Name of the operation
            error: The exception that occurred
            **kwargs: Additional information to log
        """
        log_message = f"Error in {operation}: {str(error)}"
        
        # Add additional info if provided
        if kwargs:
            additional_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            log_message += f" ({additional_info})"
        
        self.logger.error(log_message)
    
    def log_memory_summary(self, operation: str = ""):
        """
        Log a summary of GPU memory usage.
        
        Args:
            operation: Optional operation name for context
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return
            
            context = f" after {operation}" if operation else ""
            self.logger.info(f"GPU memory summary{context}:")
            
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                self.logger.info(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        except:
            self.logger.warning("Failed to log GPU memory summary")
