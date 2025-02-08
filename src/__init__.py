import os
import sys
import logging
from datetime import datetime

# Define the logging format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Create logs directory if it doesn't exist
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Create a unique log filename using the current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"running_log_{timestamp}.log"
log_filepath = os.path.join(log_dir, log_filename)

# Configure the logging
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        # Uncomment the next line to also log to the console
        # logging.StreamHandler(sys.stdout)
    ]
)

# Create a logger
logger = logging.getLogger("Cancer_Rag")