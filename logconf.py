import logging
from pathlib import Path
from datetime import datetime
def setup_logging(logs_dir: Path = Path('logs')):
    """
    Sets up logging for the application with plain-text formatting.
    """
    # Generate folder name based on current datetime
    run_dt = datetime.now().strftime('%b%d-%H%M').lower()
    run_dir = logs_dir / run_dt

    # Ensure the run directory exists
    run_dir.mkdir(parents=True, exist_ok=True)

    # Define log format
    log_format = '%(asctime)s %(name)s %(levelname)s %(message)s'
    formatter = logging.Formatter(log_format)

    # Main logger setup
    main_logger = logging.getLogger('main')
    main_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    if main_logger.hasHandlers():
        main_logger.handlers.clear()

    # Add file handler to the main logger
    main_log_file = run_dir / "main.log"
    main_file_handler = logging.FileHandler(main_log_file)
    main_file_handler.setFormatter(formatter)
    main_logger.addHandler(main_file_handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    main_logger.addHandler(console_handler)

    # Prevent log propagation
    main_logger.propagate = False

    # Function to add a FileHandler and StreamHandler to a specific logger
    def add_file_and_stdout_handler(logger_name: str, level: int):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # FileHandler
        log_file = run_dir / f"{logger_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # StreamHandler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Add both handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        # Prevent propagation to the root logger
        logger.propagate = False

    # Setup VectorIndexer logger
    add_file_and_stdout_handler('VectorIndexer', logging.INFO)

    # Function to configure a logger to log only to a file (e.g., unstructured)
    def add_file_only_handler(logger_name: str, level: int):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # FileHandler
        log_file = run_dir / f"{logger_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # Add the handler
        logger.addHandler(file_handler)

        # Prevent propagation to the root logger
        logger.propagate = False

    # Setup unstructured logger to log only to a file
    add_file_only_handler('unstructured', logging.INFO)

    # Log the run time in the main log
    main_logger.info(f"Log folder: {run_dir}")

    return run_dt, run_dir
