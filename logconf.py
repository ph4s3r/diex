import logging
from pathlib import Path
from datetime import datetime


def setup_logging(logs_dir: Path = Path('logs')):

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

    if main_logger.hasHandlers():
        main_logger.handlers.clear()

    main_log_file = run_dir / "main.log"
    main_file_handler = logging.FileHandler(main_log_file)
    main_file_handler.setFormatter(formatter)
    main_logger.addHandler(main_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    main_logger.addHandler(console_handler)

    # prevent propagation to the root logger
    main_logger.propagate = False

    # function to add a FileHandler and StreamHandler to a specific logger (double logger)
    def add_file_and_stdout_handler(logger_name: str, level: int):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        if logger.hasHandlers():
            logger.handlers.clear()

        log_file = run_dir / f"{logger_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        # prevent propagation to the root logger
        logger.propagate = False

    # function to configure a logger to log only to a file
    def add_file_only_handler(logger_name: str, level: int):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        if logger.hasHandlers():
            logger.handlers.clear()

        log_file = run_dir / f"{logger_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # prevent propagation to the root logger
        logger.propagate = False


    # declare loggers and their loglevels here
    add_file_and_stdout_handler('VectorIndexer', logging.INFO)
    add_file_and_stdout_handler('DocumentLoader', logging.INFO)
    add_file_only_handler('unstructured', logging.INFO)
    add_file_and_stdout_handler('Splitter', logging.INFO)
    add_file_and_stdout_handler('Embedder', logging.INFO)

    main_logger.info(f"Log folder: {run_dir}")

    return run_dt, run_dir
