# __init__.py
import os
import logging
from datetime import datetime

# Logging format
LOG_FORMAT = "[%(asctime)s - %(levelname)s - %(name)s - %(message)s]"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"running_log_{timestamp}.log"
log_filepath = os.path.join(LOG_DIR, log_filename)

# Create handlers
file_handler = logging.FileHandler(log_filepath)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Optional: also log to console (uncomment if needed)
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Get root logger and clear any default handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.addHandler(file_handler)
# Optional: add console logging
# root_logger.addHandler(console_handler)

# Optionally, set a specific logger for your project
logger = logging.getLogger("CancerRAG")
# No need to add handlers, will inherit from root_logger

# Optional: also ensure Chainlit logger inherits handlers
import logging
chainlit_logger = logging.getLogger("chainlit")
chainlit_logger.propagate = True
