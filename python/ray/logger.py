"""
Logging configuration for the Ray training framework.

This module configures logging for both the main process and Ray actor processes.
Import this module to initialize logging automatically.
"""

import logging
import os

# Track if logging has been initialized in this process
_logging_initialized = False


def setup_logging(log_file: str = None, force: bool = False):
    """
    Configure process-wide logging with concise time-stamped output.
    This function is idempotent - it will only set up logging once per process
    unless force=True.

    Args:
        log_file: Optional path to log file. If None, uses environment variable
                 RAY_TRAIN_LOG_FILE or defaults to /tmp/ray_train.log
        force: If True, reconfigure logging even if already initialized
    """
    global _logging_initialized

    # Skip if already initialized in this process (unless forced)
    if _logging_initialized and not force:
        return

    if log_file is None:
        # Check for environment variable first (set by main process)
        log_file = os.environ.get("RAY_TRAIN_LOG_FILE")

        if log_file is None:
            # Fallback: use /tmp for cross-process accessibility
            log_file = "/tmp/ray_train.log"

    # Ensure log file path is absolute
    log_file = os.path.abspath(log_file)

    # Create parent directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(console_formatter)
    root_logger.addHandler(ch)

    # File handler with unbuffered writes for multi-process logging
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - PID:%(process)d - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(file_formatter)
    # Force flush after each log to ensure multi-process writes are visible
    root_logger.addHandler(fh)

    # Force immediate flush for all handlers
    for handler in root_logger.handlers:
        handler.flush()

    root_logger.info(f"Logging initialized. Log file: {log_file}")

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"

    # Mark as initialized
    _logging_initialized = True


def get_logger(name: str):
    """
    Get a logger instance for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Note: Do NOT call setup_logging() here at module level, because the
# RAY_TRAIN_LOG_FILE environment variable may not be set yet during import.
# Instead, setup_logging() is called:
# - In the main process: after setting RAY_TRAIN_LOG_FILE in train_ray.py
# - In Ray actor processes: in RayActor.__init__()
