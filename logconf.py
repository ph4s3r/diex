import logging
from pathlib import Path
from datetime import datetime

# ANSI color codes
RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[94m",      # Blue
    "INFO": "\033[92m",       # Green
    "WARNING": "\033[93m",    # Yellow
    "ERROR": "\033[91m",      # Red
    "CRITICAL": "\033[95m",   # Magenta
}

class ColorFormatter(logging.Formatter):
    def __init__(self, fmt, use_colors=True):
        super().__init__(fmt)
        self.use_colors = use_colors

    def format(self, record):
        if self.use_colors:
            level_color = COLORS.get(record.levelname, RESET)
            record.msg = f"{level_color}{record.msg}{RESET}"
        return super().format(record)


def setup_logging(logs_dir: Path = Path('logs')):
    # Generate folder name based on current datetime
    run_dt = datetime.now().strftime('%b%d-%H%M').lower()
    run_dir = logs_dir / run_dt

    # Ensure the run directory exists
    run_dir.mkdir(parents=True, exist_ok=True)

    # Log formats
    log_format = '%(asctime)s %(name)s %(levelname)s %(message)s'
    file_formatter = logging.Formatter(log_format)  # Plain formatter for files
    console_formatter = ColorFormatter(log_format)  # Colorized formatter for stdout

    # Main logger setup
    main_logger = logging.getLogger('main')
    main_logger.setLevel(logging.INFO)

    if main_logger.hasHandlers():
        main_logger.handlers.clear()

    # File handler (plain text)
    main_log_file = run_dir / "main.log"
    main_file_handler = logging.FileHandler(main_log_file)
    main_file_handler.setFormatter(file_formatter)
    main_logger.addHandler(main_file_handler)

    # Console handler (colorized)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    main_logger.addHandler(console_handler)

    main_logger.propagate = False

    # Function to add a FileHandler and StreamHandler to a specific logger
    def add_file_and_stdout_handler(logger_name: str, level: int):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        if logger.hasHandlers():
            logger.handlers.clear()

        # File handler (plain text)
        log_file = run_dir / f"{logger_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)

        # Console handler (colorized)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        logger.propagate = False

    # Function to configure a logger to log only to a file
    def add_file_only_handler(logger_name: str, level: int):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        if logger.hasHandlers():
            logger.handlers.clear()

        # File handler (plain text)
        log_file = run_dir / f"{logger_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)
        logger.propagate = False

    # Declare loggers and their loglevels here
    add_file_and_stdout_handler('VectorIndexer', logging.INFO)
    add_file_and_stdout_handler('DocumentLoader', logging.INFO)
    add_file_only_handler('unstructured', logging.INFO)
    add_file_and_stdout_handler('Splitter', logging.INFO)
    add_file_and_stdout_handler('Embedder', logging.INFO)
    add_file_and_stdout_handler('Inserter', logging.INFO)

    main_logger.info(f"Log folder: {run_dir}")

    return run_dt, run_dir
