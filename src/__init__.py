import os
import logging
from datetime import datetime

# Define a standard logging format.
LOG_FORMAT = "[%(asctime)s - %(levelname)s - %(module)s - %(message)s]"

# Set up the logs directory.
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create a unique log file name based on the current timestamp.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"running_log_{timestamp}.log"
log_filepath = os.path.join(LOG_DIR, log_filename)
# Configure logging to file.
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(log_filepath),
        # Uncomment the next line to also log to the console
        # logging.StreamHandler(sys.stdout)
    ]
)

# Create a project logger.
logger = logging.getLogger("CancerRAG")