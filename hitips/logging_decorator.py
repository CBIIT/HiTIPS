import logging
import functools

def log_errors(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise  # Re-raise the exception to handle it outside if necessary
        return wrapper
    return decorator