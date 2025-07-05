import logging
import time

LOGGER = logging.getLogger("project_logger")  # Use the global logger


def log_execution_time(func):
    """Decorator to log function execution time."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Execute the function
        elapsed_time = time.time() - start_time

        function_name = func.__name__  # Get the actual function name
        LOGGER.info(f"{function_name},SUCCESS,{elapsed_time:.4f} sec")
        return result  # Return function result

    return wrapper
