import logging


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
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s,%(levelname)s,%(funcName)s,%(message)s",
        filemode="a",
    )

    # Create a global logger instance
    logger = logging.getLogger("project_logger")
    logger.setLevel(logging.INFO)
    logger.addFilter(HTTPFilter())

    # Disable Uvicorn HTTP access logs
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("uvicorn.error").disabled = True
    logging.getLogger("fastapi").disabled = True

    return logger
