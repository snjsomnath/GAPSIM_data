import logging
import time
import os

# Setup a Global constant to either enable or disable logging
ENABLE_LOGGING = True
ascii_art = """
    _____
    /     \\
    vvvvvvv  /|__/|
         I   /O,O   |
            I /_____   |      /|/|
              I       I  |      /O,O|
                I       I  |     |_____/
                    I       I  |       |      ______________
                    ~~~~~~~~~~~~~~~~
"""


def setup_logging(name, file_name='Tripsender.log', enable_logging=True):
    logger = logging.getLogger(name)
    if enable_logging:
        if not logger.handlers:
            logger.setLevel(logging.INFO)

            # Console Handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # File Handler
            if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
                # If the file doesn't exist or is empty, add the ASCII art
                with open(file_name, 'w') as f:
                    f.write(ascii_art)
            file_handler = logging.FileHandler(file_name, mode='a')  # Open the file in append mode
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    else:
        logger.disabled = True

    return logger

class MyFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            formatted_time = time.strftime(datefmt, ct)
            return formatted_time[:-3]  # Truncate microseconds to milliseconds
        else:
            return super().formatTime(record, datefmt)

# Initialize Logging
logger_name = "Tripsender.Timer"
logger = setup_logging(logger_name)
class Timer:
    """
    A simple timer class for profiling.
    """
    def __init__(self):
        self.start_time = None

    def start(self, enable_profiling):
        if enable_profiling:
            self.start_time = time.time()

    def end(self, function_name, enable_profiling):
        if enable_profiling:
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            logger.info("\n                                  {} took \n                                  {:.4f}ms to complete".format(function_name, round(elapsed_time * 1000, 4)))
