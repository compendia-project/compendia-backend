import logging
import os


class HTTPFilter(logging.Filter):
    """Custom filter to exclude HTTP request logs."""

    def filter(self, record):
        return (
            "HTTP Request:" not in record.getMessage()
            and "uvicorn.access" not in record.name
        )


def setup_logging(log_path):
    """
    Set up logging configuration with custom HTTP filtering.

    Args:
        log_path (str): Path to the log file

    Returns:
        logging.Logger: Configured logger instance
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Get the global logger instance
    logger = logging.getLogger("project_logger")
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set logger level
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s,%(levelname)s,%(funcName)s,%(message)s")
    file_handler.setFormatter(formatter)
    
    # Add filter to file handler
    file_handler.addFilter(HTTPFilter())
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    # Disable Uvicorn HTTP access logs
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("uvicorn.error").disabled = True
    logging.getLogger("fastapi").disabled = True

    return logger
